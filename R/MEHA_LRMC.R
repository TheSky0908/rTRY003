#' Low Rank Matrix Completion Based on MEHA
#' @description For low rank matrix completion task, suppose that for matrix M of
#'     n by n, we observe some entries and do not have access to the rest. We
#'     denote the row features X of n by p and column features as Z of n by p.
#'     We model the matrix as the sum of a low rank effect \code{Q} and a
#'     linear combination of the row features and the column features. Denote
#'     the coefficients of row features X and column features Z by \code{a} and
#'     \code{b}. $M = X a 1^T + Z b 1^T + Q$.
#'     The purpose of this function is to determine the optimal feature
#'     coefficients (i.e., \code{a} and \code{b}), low rank effect matrix \code{Q}
#'     and the hyperparameters (penalty strength) \code{x} based on the input
#'     training and validation sets using MEHA.
#'
#'
#' @param M_val Input matrix for validation.
#' @param M_tr Input matrix for training.
#' @param M_val_index Index of validation entries.
#' @param M_tr_index Index of training entries.
#' @param A Row feature matrix of n by p.
#' @param B Column feature matrix of n by p.
#' @param group A vector to describe the feature group information, with each
#'     element representing the specific number of features in each group.
#' @param N Total iterations. Default is 300.
#' @param alpha Default is 1e-4.
#' @param beta Default is 1e-4.
#' @param eta Default is 1e-4.
#' @param gamma Default is 10.
#' @param c Default is 2
#' @param c_p Default is 0.48.
#' @param auto_tuning Whether an auto-hyperparameter-tuning is needed.Default is
#'     \code{FALSE}.
#' @param temperature Temperature of simulating annealing method for auto-
#'     hyperparameter-tuning. Default is 0.1.
#'
#' @return
#'
#'   \item{x}{A vector of length (2M+1), where M denotes the total group number.
#'       The first M values are the within-group penalty strengths of row feature
#'       coefficients \code{a}, the second M values are within-group penalty strength
#'       of column feature coefficients \code{b}, the last value is the penalty
#'       strengths of low rank effect matrix \code{Q}.}
#'   \item{a}{Row feature coefficient.}
#'   \item{b}{Column feature coefficient.}
#'   \item{Q}{low rank effect matrix.}
#'   \item{theta_a}{to}
#'   \item{theta_b}{to}
#'   \item{theta_Q}{to}
#'   \item{Xconv}{Describe the relative convergence situation of sequence X,
#'       based on l2-norm.}
#'   \item{Yconv}{Describe the relative convergence situation of sequence Y,
#'       based on l2-norm.}
#'   \item{Thetaconv}{Describe the relative convergence situation of sequence theta,
#'       based on l2-norm.}
#'   \item{Fseq}{The upper function value in each iteration.}
#'
#'
#' @export
#'

MEHA_LRMC = function(M_val, M_tr, M_val_index, M_tr_index,
                     A, B, group, N = 200, alpha = 1e-4, beta = 1e-4, eta = 1e-4,
                     gamma = 10, c = 2, c_p = 0.48, auto_tuning = FALSE,
                     temperature = 0.1){

  library(progress)
  library(truncnorm)

  main_fun <- function(M_val, M_tr, M_val_index, M_tr_index, A, B, group, N, alpha, beta, eta, gamma, c, c_p){

    G_num = dim(group)[1]
    n = dim(M_val_index)[1]
    p = dim(A)[2]
    if (p != sum(group)) {
      return(print("Error: p != sum(group), he grouping condition contradicts the number of features"))
    }

    e_1n = matrix(rep(1), nrow = n)
    e_1p = matrix(rep(1), nrow = p)
    E = matrix(rep(1), nrow = n, ncol = n)
    e0_2g1 = matrix(rep(0), nrow = 2*G_num + 1)
    e0_g = matrix(rep(0), nrow = G_num)
    e_0p = matrix(rep(0), nrow = p)

    # initial values
    a =  matrix(rnorm(p), nrow = p)
    b =  matrix(rnorm(p), nrow = p)
    Q =  matrix(rnorm(n*n), nrow = n, ncol = n)
    x =  matrix(rep(1), nrow = 2*G_num + 1)
    theta_a = matrix(rep(1), nrow = p)
    theta_b = matrix(rep(1), nrow = p)
    theta_Q = E


    # objective function
    up_fun = function(x, a, b, Q){
      result = 0.5*norm( (M_val - A %*% a %*% t(e_1n) - t(B %*% b %*% t(e_1n)) - Q) * M_val_index, type = "F")^2
      return(result)
    }

    low_fun = function(x, a, b, Q){
      result = 0.5*norm( (M_tr - A %*% a %*% t(e_1n) - t(B %*% b %*% t(e_1n)) - Q) * M_tr_index, type = "F")^2
      return(result)
    }




    # update function
    F_x = function(x, a, b, Q){
      result = e0_2g1
      return(result)
    }

    F_a = function(x, a, b, Q){
      result = e_0p
      for (i in 1:nrow(M_val_index)) {
        for (j in 1:ncol(M_val_index)) {
          if (M_val_index[i, j] == 1) {  # 判断是否需要加入求和
            result = result + as.numeric((M_val[i, j] - A[i, ] %*% a - B[j, ] %*% b - Q[i,j] ))*(-1*A[i, ])
          }
        }
      }
      return(result)
    }

    F_b = function(x, a, b, Q){
      result = e_0p
      for (i in 1:nrow(M_val_index)) {
        for (j in 1:ncol(M_val_index)) {
          if (M_val_index[i, j] == 1) {  # 判断是否需要加入求和
            result = result + as.numeric((M_val[i, j] - A[i, ] %*% a - B[j, ] %*% b - Q[i,j] ))*(-1*B[j, ])
          }
        }
      }
      return(result)
    }

    F_Q = function(x, a, b, Q){
      result = 0*E
      for (i in 1:nrow(M_val_index)) {
        for (j in 1:ncol(M_val_index)) {
          if (M_val_index[i, j] == 1) {  # 判断是否需要加入求和
            result = result + as.numeric((M_val[i, j] - A[i, ] %*% a - B[j, ] %*% b - Q[i,j] ))*(-1*E)
          }
        }
      }
      return(result)
    }



    f_x = function(x, a, b, Q){
      result = e0_2g1
      return(result)
    }


    f_a = function(x, a, b, Q){
      result = e_0p
      for (i in 1:nrow(M_tr_index)) {
        for (j in 1:ncol(M_tr_index)) {
          if (M_tr_index[i, j] == 1) {  # 判断是否需要加入求和
            result = result + as.numeric((M_tr[i, j] - A[i, ] %*% a - B[j, ] %*% b - Q[i,j] ))*(-1*A[i, ])
          }
        }
      }
      return(result)
    }

    f_b = function(x, a, b, Q){
      result = e_0p
      for (i in 1:nrow(M_tr_index)) {
        for (j in 1:ncol(M_tr_index)) {
          if (M_tr_index[i, j] == 1) {  # 判断是否需要加入求和
            result = result + as.numeric((M_tr[i, j] - A[i, ] %*% a - B[j, ] %*% b - Q[i,j] ))*(-1*B[j, ])
          }
        }
      }
      return(result)
    }

    f_Q = function(x, a, b, Q){
      result = 0*E
      for (i in 1:nrow(M_tr_index)) {
        for (j in 1:ncol(M_tr_index)) {
          if (M_tr_index[i, j] == 1) {  # 判断是否需要加入求和
            result = result + as.numeric((M_tr[i, j] - A[i, ] %*% a - B[j, ] %*% b - Q[i,j] ))*(-1*E)
          }
        }
      }
      return(result)
    }


    g_x = function(x, a, b, Q){
      resulta = e0_g
      resultb = e0_g
      for (k in 1:G_num) {
        ak = a[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        resulta[k] = norm( ak ,type = "2")
      }
      for (k in 1:G_num) {
        bk = b[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        resultb[k] = norm( bk ,type = "2")
      }
      resultQ = norm( Q ,type = "F")
      return(rbind(resulta, resultb,resultQ))
    }


    # proximal operator
    prox_eta_a = function(x, a, b, Q, theta_a, theta_b, theta_Q){
      z_a = theta_a - eta * (f_a(x, theta_a, theta_b, theta_Q) + (theta_a - a) / gamma)
      result = e_0p
      for (k in 1:G_num) {
        z_ak = z_a[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        if (eta * x[k] > 0) {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = (1 - eta * x[k]/(max(norm(z_ak,type = "2"), eta * x[k])))*z_ak
        } else {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = z_ak
        }
      }
      return(result)
    }


    prox_eta_b = function(xx, a, b, Q, theta_a, theta_b, theta_Q){
      z_b = theta_b - eta * (f_b(x, theta_a, theta_b, theta_Q) + (theta_b - b) / gamma)
      result = e_0p
      for (k in 1:G_num) {
        z_bk = z_b[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        if (eta * x[k + G_num] > 0) {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = (1 - eta * x[k + G_num]/(max(norm(z_bk,type = "2"), eta * x[k + G_num])))*z_bk
        } else {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = z_bk
        }
      }
      return(result)
    }

    prox_eta_Q = function(x, a, b, Q, theta_a, theta_b, theta_Q){
      z_Q = theta_Q - eta * (f_Q(x, theta_a, theta_b, theta_Q) + (theta_Q - Q) / gamma)
      result = E
      svd_result = svd(Q)
      if (eta * x[2*G_num + 1] > 0) {
        D = pmax(svd_result$d - beta * x[2*G_num + 1], 0)
      } else {
        D = 0*svd_result$d
      }
      result = svd_result$u %*% diag(D) %*% t(svd_result$v)
      return(result)
    }




    prox_beta_a = function(x, a, b, Q, dka, dkb, dkQ){
      z_a = a - beta * dka
      result = e_0p
      for (k in 1:G_num) {
        z_ak = z_a[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        if (beta * x[k] > 0) {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = (1 - beta * x[k]/(max(norm(z_ak,type = "2"), eta * x[k])))*z_ak
        } else {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = z_ak
        }
      }
      return(result)
    }


    prox_beta_b = function(x, a, b, Q, dka, dkb, dkQ){
      z_b = b - beta * dkb
      result = e_0p
      for (k in 1:G_num){
        z_bk = z_b[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        if (beta * x[k + G_num] > 0) {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = (1 - beta * x[k + G_num]/(max(norm(z_bk,type = "2"), beta * x[k + G_num])))*z_bk
        } else {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = z_bk
        }
      }
      return(result)
    }

    prox_beta_Q = function(x, a, b, Q, dka, dkb, dkQ){
      z_Q = Q - beta * dkQ
      result = E
      svd_result = svd(Q)
      if (beta * x[2*G_num + 1] > 0) {
        D = pmax(svd_result$d - beta * x[2*G_num + 1], 0)
      } else {
        D = 0*svd_result$d
      }
      result = svd_result$u %*% diag(D) %*% t(svd_result$v)
      return(result)
    }


    ## algorithm
    arrF = numeric(N)
    res1 = numeric(N)
    res2 = numeric(N)
    res3 = numeric(N)

    for (k in 1:N) {
      xk = x
      ak = a
      bk = b
      Qk = Q
      theta_ak = theta_a
      theta_bk = theta_b
      theta_Qk = theta_Q
      # ck = 0.49
      ck = c*(k + 1)^c_p
      theta_a = prox_eta_a(x, a, b, Q, theta_a, theta_b, theta_Q)
      theta_b = prox_eta_b(x, a, b, Q, theta_a, theta_b, theta_Q)
      theta_Q = prox_eta_Q(x, a, b, Q, theta_a, theta_b, theta_Q)

      dkx = (1/ck) * F_x(x, a, b, Q) + f_x(x, a, b, Q) + g_x(x, a, b, Q) - f_x(x, theta_a, theta_b, theta_Q) - g_x(x, theta_a, theta_b, theta_Q)

      x = pmax(x - alpha * dkx,e0_2g1)

      dka = (1/ck) * F_a(x, a, b, Q) + f_a(x, a, b, Q) - (a - theta_a)/gamma

      dkb = (1/ck) * F_b(x, a, b, Q) + f_b(x, a, b, Q) - (b - theta_b)/gamma

      dkQ = (1/ck) * F_Q(x, a, b, Q) + f_Q(x, a, b, Q) - (Q - theta_Q)/gamma


      a = prox_beta_a(x, a, b, Q, dka, dkb, dkQ)

      b = prox_beta_b(x, a, b, Q, dka, dkb, dkQ)

      Q = prox_beta_Q(x, a, b, Q, dka, dkb, dkQ)



      res1[k] = norm(x - xk , "2") / norm(xk, "2")
      res2[k] = (norm(a - ak, "2") + norm(b - bk, "2") + norm(Q - Qk, "F")) / (norm(a, "2") + norm(b, "2") + norm(Q, "F"))
      res3[k] = (norm(theta_a - theta_ak, "2") + norm(theta_b - theta_bk, "2") + norm(theta_Q - theta_Qk, "F")) / (norm(theta_a, "2") + norm(theta_b, "2") + norm(theta_Q, "F"))
      arrF[k] = up_fun(x, a, b, Q)
    }

    return(list(x = x, a = a, b = b, Q = Q, theta_a = theta_a,theta_B = theta_b, theta_Q = theta_Q, Xconv = res1, Yconv = res2, Thetaconv = res3, Fseq = arrF))


  }

  if(auto_tuning == TRUE){
    message("\n","Auto-hyperparameters-tuning is proceeding now.")

    iter <- 100
    T <- temperature

    pb <- progress_bar$new(
      total = iter,
      format = "  Finished :current/:total [:bar] :percent  remaining time :eta"
    )


    alpha.seq <- numeric(iter)
    beta.seq <- numeric(iter)
    eta.seq <- numeric(iter)
    value <- numeric(iter)

    alpha.seq[1] <- alpha
    beta.seq[1] <- beta
    eta.seq[1] <- eta

    result <- main_fun(M_val, M_tr, M_val_index, M_tr_index, A, B, group, N, alpha = alpha.seq[1], beta = beta.seq[1], eta = eta.seq[1], gamma = gamma, c = c, c_p = c_p)
    value[1] <- result$Fseq[order(result$Fseq, decreasing = FALSE)[1]]


    set.seed(123)
    for (j in 2:iter) {
      #T <- T*exp(-0.01*j)
      alpha.seq[j] <- rtruncnorm(n = 1, a = 0, mean = alpha.seq[j-1], sd = 1e-3)
      beta.seq[j] <- rtruncnorm(n = 1, a = 0, mean = beta.seq[j-1], sd = 1e-6)
      eta.seq[j] <- rtruncnorm(n = 1, a = 0, mean = eta.seq[j-1], sd = 1e-6)
      result <-  main_fun(M_val, M_tr, M_val_index, M_tr_index, A, B, group, N, alpha = alpha.seq[j], beta = beta.seq[j], eta = eta.seq[j],gamma = gamma, c = c, c_p = c_p)
      candidate <- result$Fseq[order(result$Fseq, decreasing = FALSE)[1]]
      if(candidate > value[j-1] & runif(n = 1) > exp((value[j-1]-candidate)/T)){
        value[j] <- value[j-1]
      } else {
        value[j] <- candidate
      }
      pb$tick()
    }


    opt_index <- order(value)[1]

    cat("\n", "Auto-hyperparameters-tuning is done.")
    cat("\nFinal hyper-paramaters (alpha,beta,eta) are chosen as:",c(alpha.seq[opt_index], beta.seq[opt_index], eta.seq[opt_index]))

    return(main_fun(M_val, M_tr, M_val_index, M_tr_index, A, B, group, N, alpha = alpha.seq[opt_index], beta = beta.seq[opt_index], eta = eta.seq[opt_index],gamma = gamma, c = c, c_p = c_p))

  }
  else{
    main_fun(M_val, M_tr, M_val_index, M_tr_index, A, B, group, N, alpha, beta, eta,gamma = gamma, c = c, c_p = c_p)
  }


}
