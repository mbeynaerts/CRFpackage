#include <cmath>
#include <Rcpp.h>

using namespace Rcpp;


// [[Rcpp::export]]
IntegerMatrix IndGreater(NumericVector &x) {
  int n = x.size();
  IntegerMatrix elem(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      if (x[j] >= x[i]) {
        elem(j,i) = 1;
      } else {
        elem(j,i) = 0;
      }
    }
  }
  return elem;
}

// [[Rcpp::export]]
IntegerMatrix IndLess(NumericVector &x) {
  int n = x.size();
  IntegerMatrix elem(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      if (x[j] <= x[i]) {
        elem(j,i) = 1;
      } else {
        elem(j,i) = 0;
      }
    }
  }
  return elem;
}

// [[Rcpp::export]]
IntegerMatrix IndEqual(NumericVector &x) {
  int n = x.size();
  IntegerMatrix elem(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      if (x[j] == x[i]) {
        elem(j,i) = 1;
      } else {
        elem(j,i) = 0;
      }
    }
  }
  return elem;
}

// [[Rcpp::export]]
int Ind2(NumericVector &x, NumericVector &y, double &a, double &b) {
  int n = x.size();
  int sum = 0;
  for (int i=0; i<n; i++) {
    if (x[i] >= a and y[i] >= b) {
      sum += 1;
    } else {
      sum += 0;
    }
  }
  return sum;
}

// [[Rcpp::export]]
NumericMatrix risksetC(NumericVector &x, NumericVector &y) {
  int n = x.size();
  NumericMatrix riskset(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      riskset (j,i) = Ind2(x, y, x[j], y[i]);
    }
  }
  return riskset;
}

// [[Rcpp::export]]
IntegerMatrix DeltaC(NumericVector &x, NumericVector &y) {
  int n = x.size();
  IntegerMatrix delta(n);
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; i++) {
      delta (j,i) = x[j]*y[i];
    }
  }
  return delta;
}



// [[Rcpp::export]]
double logLikC(const NumericVector &riskset1,
               const NumericVector &riskset2,
               const NumericVector &logtheta1,
               const NumericVector &logtheta2,
               const NumericVector &delta1,
               const NumericVector &delta2,
               const NumericVector &I1,
               const NumericVector &I2,
               const NumericVector &I3,
               const NumericVector &I4,
               const NumericVector &I5,
               const NumericVector &I6,
               const NumericVector &w1,
               const NumericVector &w2) {

  double sum1;
  double sum2;

  sum1 = Rcpp::sum(w1*delta1*I1*(logtheta1*I5 - Rcpp::log(riskset1 + I2*Rcpp::exp(logtheta1) - I2)));
  sum2 = Rcpp::sum(w2*delta2*I3*(logtheta2*I6 - Rcpp::log(riskset2 + I4*Rcpp::exp(logtheta2) - I4)));

  return(-sum1-sum2);
}

// [[Rcpp::export]]
NumericVector gradientC(const NumericVector &riskset1,
                        const NumericVector &riskset2,
                        const NumericVector &logtheta1,
                        const NumericVector &logtheta2,
                        const Rcpp::List &deriv,
                        const int &df,
                        const NumericVector &delta1,
                        const NumericVector &delta2,
                        const NumericVector &I1,
                        const NumericVector &I2,
                        const NumericVector &I3,
                        const NumericVector &I4,
                        const NumericVector &I5,
                        const NumericVector &I6,
                        const NumericVector w1,
                        const NumericVector w2) {

  int n = riskset1.length();
  int totalparam = df*df;
  NumericVector result(totalparam);
  NumericVector common1(n);
  NumericVector common2(n);

  /* Transform list of derivative matrices into vector of matrices */
  std::vector<NumericMatrix> deriv_vec(totalparam);
  for (int k = 0; k < totalparam; ++k) {
    NumericMatrix deriv_R = deriv[k];
    deriv_vec[k] = deriv_R;
  }


  common1 = w1*delta1*I1*(I5 - I2*Rcpp::exp(logtheta1)/(riskset1 + I2*Rcpp::exp(logtheta1) - I2));
  common2 = w2*delta2*I3*(I6 - I4*Rcpp::exp(logtheta2)/(riskset2 + I4*Rcpp::exp(logtheta2) - I4));

  for (int m=0; m<totalparam; m++) {

    double sum1 = 0.0;
    double sum2 = 0.0;

    /* Calculation of L1 */
    for (int j=0; j<n; j++) {
      sum1 += common1(j)*deriv_vec[m](j,0);
      sum2 += common2(j)*deriv_vec[m](j,1);
    }

    result(m) = -sum1-sum2;

  }

return(result);

}

// [[Rcpp::export]]
NumericVector gradientNew(const arma::colvec &riskset1,
                          const arma::colvec &riskset2,
                          const arma::colvec &logtheta1,
                          const arma::colvec &logtheta2,
                          const arma::colvec &delta1,
                          const arma::colvec &delta2,
                          const arma::colvec &I1,
                          const arma::colvec &I2,
                          const arma::colvec &I3,
                          const arma::colvec &I4,
                          const arma::colvec &I5,
                          const arma::colvec &I6,
                          const arma::mat &X1,
                          const arma::mat &X2,
                          const arma::uvec &idxN1,
                          const arma::uvec &idxN2,
                          const arma::colvec& w1,
                          const arma::colvec& w2) {

  int K = X1.n_rows;
  int n = riskset1.size();
  int totalparam = pow(X1.n_cols, 2);
  NumericVector result(totalparam);
  arma::colvec common1(n);
  arma::colvec common2(n);

  common1 = w1 % delta1 % I1 % (I5 - I2 % arma::exp(logtheta1)/(riskset1 + I2 % arma::exp(logtheta1) - I2));
  common2 = w2 % delta2 % I3 % (I6 - I4 % arma::exp(logtheta2)/(riskset2 + I4 % arma::exp(logtheta2) - I4));

  arma::mat deriv_mat(K,K), deriv_mat_t(K,K);
  arma::colvec deriv1(deriv_mat_t.memptr(), deriv_mat_t.n_elem, false, true);
  arma::colvec deriv2(deriv_mat.memptr(), deriv_mat.n_elem, false, true);

  for (int m=0; m<totalparam; m++) {

    int idx1 = totalparam % m;
    int idx2 = totalparam / m;

    deriv_mat = arma::kron(X1.col(idx1), X2.col(idx2).as_row());
    deriv_mat_t = deriv_mat.t();

    double sum1 = arma::accu(common1 % deriv1.elem(idxN1));
    double sum2 = arma::accu(common2 % deriv2.elem(idxN2));

    result(m) = -sum1-sum2;

  }

  return(result);

}


// [[Rcpp::export]]
NumericVector gradientPoly(const NumericVector &riskset1,
                           const NumericVector &riskset2,
                           const NumericVector &logtheta1,
                           const NumericVector &logtheta2,
                           const Rcpp::List &deriv,
                           const int &df,
                           const NumericVector &delta1,
                           const NumericVector &delta2,
                           const NumericVector &I1,
                           const NumericVector &I2,
                           const NumericVector &I3,
                           const NumericVector &I4,
                           const NumericVector &I5,
                           const NumericVector &I6) {

  int n = riskset1.length();
  NumericVector result(df);
  NumericVector common1(n);
  NumericVector common2(n);

  /* Transform list of derivative matrices into vector of matrices */
  std::vector<NumericMatrix> deriv_vec(df);
  for(int k = 0; k < df; ++k) {
    NumericMatrix deriv_R = deriv[k];
    /* arma::mat derivMat(deriv_R.begin(), deriv_R.rows(), deriv_R.cols(), false, true);
     deriv_vec[k] = derivMat; */
    deriv_vec[k] = deriv_R;
  }

  common1 = delta1*I1*(I5 - I2*Rcpp::exp(logtheta1)/(riskset1 + I2*Rcpp::exp(logtheta1) - I2));
  common2 = delta2*I3*(I6 - I4*Rcpp::exp(logtheta2)/(riskset2 + I4*Rcpp::exp(logtheta2) - I4));

  for (int m=0; m<df; m++) {

    double sum1 = 0.0;
    double sum2 = 0.0;

    /* Calculation of L1 */
    for (int j=0; j<n; j++) {
      sum1 += common1(j)*deriv_vec[m](j,0);
      sum2 += common2(j)*deriv_vec[m](j,1);
    }

    result(m) = -sum1-sum2;

  }

  return(result);

}


// [[Rcpp::export]]
NumericMatrix hessianC(const NumericVector &riskset1,
                       const NumericVector &riskset2,
                       const NumericVector &logtheta1,
                       const NumericVector &logtheta2,
                       const Rcpp::List &deriv,
                       const NumericVector &delta1,
                       const NumericVector &delta2,
                       const int df,
                       const NumericVector &I1,
                       const NumericVector &I2,
                       const NumericVector &I3,
                       const NumericVector &I4,
                       const NumericVector &w1,
                       const NumericVector &w2) {

  int n = riskset1.length();
  int totalparam = df*df;
  NumericVector common1(n);
  NumericVector common2(n);

  NumericMatrix result(totalparam);

  /* Transform list of derivative matrices into vector of matrices */
  std::vector<NumericMatrix> deriv_vec(totalparam);
  for(int k = 0; k < totalparam; ++k) {
    NumericMatrix deriv_R = deriv[k];
    /* arma::mat derivMat(deriv_R.begin(), deriv_R.rows(), deriv_R.cols(), false, true);
    deriv_vec[k] = derivMat; */
    deriv_vec[k] = deriv_R;
  }

    common1 = -w1*delta1*I1*(riskset1 - I2)*I2*Rcpp::exp(logtheta1)/Rcpp::pow(riskset1 - I2 + I2*Rcpp::exp(logtheta1),2);
    common2 = -w2*delta2*I3*(riskset2 - I4)*I4*Rcpp::exp(logtheta2)/Rcpp::pow(riskset2 - I4 + I4*Rcpp::exp(logtheta2),2);

  for (int m = 0; m < totalparam; m++) {
    for (int l = m; l < totalparam; l++) {

      double sum1 = 0.0;
      double sum2 = 0.0;

      for (int j=0; j<n; j++) {
        sum1 += common1(j)*deriv_vec[m](j,0)*deriv_vec[l](j,0);
        sum2 += common2(j)*deriv_vec[m](j,1)*deriv_vec[l](j,1);
      }

      result(m,l) = -sum1-sum2;
      result(l,m) = result(m,l);

    }
  }

return(result);

}

// [[Rcpp::export]]
arma::mat hessianNew(const arma::colvec& riskset1,
                       const arma::colvec &riskset2,
                       const arma::colvec &logtheta1,
                       const arma::colvec &logtheta2,
                       const arma::colvec &delta1,
                       const arma::colvec &delta2,
                       const arma::colvec &I1,
                       const arma::colvec &I2,
                       const arma::colvec &I3,
                       const arma::colvec &I4,
                       const arma::mat& X1,
                       const arma::mat& X2,
                       const arma::uvec& idxN1,
                       const arma::uvec& idxN2,
                       const arma::colvec& w1,
                       const arma::colvec& w2) {

  int K = X1.n_rows;
  int n = riskset1.n_rows;
  int totalparam = pow(X1.n_cols, 2);
  arma::colvec common1(n);
  arma::colvec common2(n);

  arma::mat result(totalparam, totalparam);

  common1 = -w1 % delta1 % I1 % (riskset1 - I2) % I2 % arma::exp(logtheta1) / arma::pow(riskset1 - I2 + I2 % arma::exp(logtheta1),2);
  common2 = -w2 % delta2 % I3 % (riskset2 - I4) % I4 % arma::exp(logtheta2) / arma::pow(riskset2 - I4 + I4 % arma::exp(logtheta2),2);

  arma::mat deriv_mat(K,K), deriv_mat_t(K,K), deriv_mat_l(K,K), deriv_mat_l_t(K,K);

  arma::colvec deriv1m(deriv_mat_t.memptr(), deriv_mat_t.n_elem, false, true);
  arma::colvec deriv2m(deriv_mat.memptr(), deriv_mat.n_elem, false, true);
  arma::colvec deriv1l(deriv_mat_l_t.memptr(), deriv_mat_l_t.n_elem, false, true);
  arma::colvec deriv2l(deriv_mat_l.memptr(), deriv_mat_l.n_elem, false, true);

  for (int m = 0; m < totalparam; m++) {

    int idx1 = totalparam % m;
    int idx2 = totalparam / m;

    deriv_mat = arma::kron(X1.col(idx1), X2.col(idx2).as_row());
    deriv_mat_t = deriv_mat.t();

    for (int l = m; l < totalparam; l++) {

      idx1 = totalparam % l;
      idx2 = totalparam / l;

      deriv_mat_l = arma::kron(X1.col(idx1), X2.col(idx2).as_row());
      deriv_mat_l_t = deriv_mat_l.t();

      double sum1 = accu(common1 % deriv1m.elem(idxN1) % deriv1l.elem(idxN1));
      double sum2 = accu(common2 % deriv2m.elem(idxN2) % deriv2l.elem(idxN2));

      result(l,m) = -sum1-sum2;

    }
  }

  return arma::symmatl(result);

}


// [[Rcpp::export]]
NumericMatrix hessianPolyC(const NumericVector &riskset1,
                           const NumericVector &riskset2,
                           const NumericVector &logtheta1,
                           const NumericVector &logtheta2,
                           const Rcpp::List &deriv,
                           const int &df,
                           const NumericVector &delta1,
                           const NumericVector &delta2,
                           const NumericVector &I1,
                           const NumericVector &I2,
                           const NumericVector &I3,
                           const NumericVector &I4) {

  int n = riskset1.length();
  NumericVector common1(n);
  NumericVector common2(n);

  NumericMatrix result(df);

  /* Transform list of derivative matrices into vector of matrices */
  std::vector<NumericMatrix> deriv_vec(df);
  for(int k = 0; k < df; ++k) {
    NumericMatrix deriv_R = deriv[k];
    /* arma::mat derivMat(deriv_R.begin(), deriv_R.rows(), deriv_R.cols(), false, true);
    deriv_vec[k] = derivMat; */
    deriv_vec[k] = deriv_R;
  }

  common1 = -delta1*I1*(riskset1 - I2)*I2*Rcpp::exp(logtheta1)/Rcpp::pow(riskset1 - I2 + I2*Rcpp::exp(logtheta1),2);
  common2 = -delta2*I3*(riskset2 - I4)*I4*Rcpp::exp(logtheta2)/Rcpp::pow(riskset2 - I4 + I4*Rcpp::exp(logtheta2),2);

  for (int m = 0; m < df; m++) {
    for (int l = m; l < df; l++) {

      double sum1 = 0.0;
      double sum2 = 0.0;

      for (int j=0; j<n; j++) {
        sum1 += common1(j)*deriv_vec[m](j,0)*deriv_vec[l](j,0);
        sum2 += common2(j)*deriv_vec[m](j,1)*deriv_vec[l](j,1);
      }

      result(m,l) = -sum1-sum2;
      result(l,m) = result(m,l);

    }
  }

  return(result);

}

NumericMatrix testfunct(const List &deriv,
                        const int &df) {

  int totalparam = df*df;

  std::vector<NumericMatrix> deriv_vec(totalparam);
  for(int k = 0; k < totalparam; ++k) {
    NumericMatrix deriv_R = deriv[k];
    /* arma::mat derivMat(deriv_R.begin(), deriv_R.rows(), deriv_R.cols(), false, true);
    deriv_vec[k] = derivMat; */
    deriv_vec[k] = deriv_R;
  }
  NumericMatrix result = deriv_vec[0];
  return(result);
}

