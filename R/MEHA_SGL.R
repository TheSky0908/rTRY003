#' Sparse Group Lasso Based on MEHA
#' @description Sparse Group Lasso extends the Lasso and Group Lasso techniques
#'     by enforcing sparsity both at the group level and within each group. Sparse
#'     Group Lasso applies an L2 penalty to groups of coefficients and an L1
#'     penalty to individual coefficients, encouraging sparsity within and between
#'     groups. This approach introduces two hyperparameters: within-group penalty
#'     and group penalty which control the extent of regularization.
#'     The purpose of this function is to determine the optimal feature
#'     coefficients \code{y} and the hyperparameters \code{x} of the sparse group
#'     Lasso based on the input training and validation sets using MEHA.
#'
#'
#' @param A_val Input feature matrix of validation set, each row of which
#'     is an observation vector.
#' @param b_val Quantitative response variable of validation set.
#' @param A_tr Input feature matrix of training set.
#' @param b_tr Quantitative response variable of training set.
#' @param group A vector of length M (where M denotes the total group number) to
#'     describe the feature group information, with each element representing the
#'     specific number of features in each group.
#' @param N Total iterations. Default is 300.
#' @param alpha Default is 1e-5.
#' @param beta Default is 1e-5.
#' @param eta Default is 1e-5.
#' @param gamma Default is 1.
#' @param c Default is 1.
#' @param p Default is 0.3.
#' @param auto_tuning Whether an auto-hyperparameter-tuning is needed.
#'     Default is \code{FALSE}.
#' @param temperature Temperature of simulating annealing method for auto-
#'     hyperparameter-tuning. Default is 0.01.
#'
#' @return
#'
#'   \item{x}{A vector of length (M+1), where M denotes the total group number.
#'       The first M values are the within-group penalty strengths, while the last
#'       value represents the group penalty.}
#'   \item{y}{The feature coefficient vector, of dimension p, where p is the
#'       feature number.}
#'   \item{theta}{to}
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
#' @examples
#' library(mvtnorm)
#'
#' N <- 100 #sample size
#' p <- 600 #feature number
#' M <- 30 #feature group number
#' beta_i <- c(1:5,rep(0,195))
#' true_beta <- rep(beta_i,3) #true coefficient

#' set.seed(123)
#' a <- matrix(rnorm(3*N*p), ncol = p)
#' epsilon <- matrix(rnorm(3*N), ncol = 1) #residual
#' sigma <- norm(a %*% true_beta, type = "2") / norm(epsilon, type = "2")
#' b <- a %*% true_beta + sigma/2 * epsilon

#' SNR <- norm(a %*% true_beta, type = "2") / norm(b - a %*% true_beta,
#'  type = "2")
#' cat("SNR is", SNR) #the signal-to-noise ratio is controlled at 2
#'
#' set.seed(123)
#' #split into training, validation and testing set
#' A_val = a[1:100, ]
#' b_val = b[1:100]
#' A_tr = a[101:200, ]
#' b_tr = b[101:200]
#' A_test = a[201:300, ]
#' b_test = b[201:300]
#'
#' group <- as.matrix(rep(x = p/M, M)) #feature group information
#'
#' result = MEHA_SGL(A_val, b_val, A_tr, b_tr, group,N = 200,
#'     auto_tuning = TRUE, alpha = 3e-4)
#'
#' plot(result$Yconv,ylab = expression(paste("||",y^k - y^(k - 1),"||/||",
#' y^(k - 1),"||")), xlab = "iteration",type = "l",
#'  main = "Y Sequence Convergence")
#'
#' plot(result$Fseq, xlab = "iteration", type = "l",
#' ylab = expression(F(x^k,y^k)))
#'
#' norm(A_val %*% result$y - b_val, type = "2")/length(b_val)
#'
#'

MEHA_SGL = function(A_val, b_val, A_tr, b_tr, group, N = 300, alpha = 1e-5,
                    beta = 1e-5, eta = 1e-5, gamma = 1, c = 1, p = 0.3,
                    auto_tuning = FALSE, temperature = 0.01){

  library(progress)
  library(truncnorm)

  main_fun <- function(A_val, b_val, A_tr, b_tr, group,
                       N, alpha, beta, eta, gamma, c, p){

    p_fea = dim(A_val)[2]
    group_num = dim(group)[1]

    if (p_fea != sum(group)) {
      return(print("Error: p_fea != sum(group), he grouping condition contradicts the number of features"))
    }

    # initial value
    x = matrix(rep(5), nrow = group_num + 1)
    y = matrix(rep(10), nrow = p_fea)
    theta = matrix(rep(1), nrow = p_fea)


    e0 = matrix(rep(0), nrow = group_num + 1)
    ep = matrix(rep(0), nrow = p_fea)
    ep1 = matrix(rep(1), nrow = p_fea)


    # objective function
    up_fun = function(x,y){
      result = 0.5*norm(A_val %*% y - b_val, type = "2")^2
      return(result)
    }
    fun_group = function(x,y){
      result = 0
      for (k in 1:group_num){
        result = x[k]*norm( y[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] ,type = "2") + result
      }
      return(result)
    }

    low_fun = function(x, y){
      result = 0.5*norm(A_tr %*%  y - b_tr, type = "2")^2 + fun_group(x,y) + x[group_num + 1]* norm( y ,type = "1")
      # + x[group_num+1]* norm(y, type = "1")
      return(result)
    }


    ## update function
    F_x = function(x, y){
      result = e0
      return(result)
    }

    F_y = function(x, y){
      result = t(t(A_val %*% y - b_val) %*% A_val)
      return(result)
    }

    f_x = function(x, y){
      result = e0
      return(result)
    }

    f_y = function(x, y){
      result = t(t(A_tr %*% y - b_tr) %*% A_tr)
      return(result)
    }

    g_x = function(x, y){
      result = e0
      for (k in 1:group_num){
        yk = y[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        result[k] = norm( yk ,type = "2")
      }
      result[group_num + 1] = norm( y ,type = "1")
      return(result)
    }


    #proximal operators
    prox_eta = function(x, y, theta){
      z = theta - eta * (f_y(x, theta) + (theta - y) / gamma)
      z2 = ep
      result = ep
      for (k in 1:group_num) {
        zk = z[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        ek = ep1[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        # 内部近端算子
        if (eta * x[group_num + 1] > 0) {
          z2[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = sign(zk) * pmax((abs(zk) - eta * x[group_num + 1] * ek), 0*ek)
        }
        else{
          z2[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = zk
        }
        z2k = z2[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        # 外部近端算子
        if (eta * x[k] > 0) {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = (1 - eta * x[k]/(max(norm(z2k,type = "2"), eta * x[k])))*z2k
        }
        else {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = z2k
        }
      }
      return(result)
    }

    prox_beta = function(x, y, dky){
      z = y - beta * dky
      z2 = ep
      result = ep
      for (k in 1:group_num) {
        zk = z[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        ek = ep1[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]

        # 内部近端算子
        #print(c(k,beta,x[group_num+1]))

        if (beta * x[group_num + 1] > 0) {
          z2[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = sign(zk) * pmax((abs(zk) - eta * x[group_num + 1] * ek), 0*ek)
        }
        else{
          z2[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = zk
        }
        z2k = z2[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])]
        # 外部近端算子
        if (beta * x[k] > 0) {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = (1 - beta * x[k]/(max(norm(z2k,type = "2"), beta * x[k])))*z2k
        }
        else {
          result[(sum(group[1:k]) - group[k] + 1):sum(group[1:k])] = z2k
        }
      }
      return(result)
    }


    # array to store the results
    arrF = numeric(N)
    res1 = numeric(N)
    res2 = numeric(N)
    res3 = numeric(N)

    # iteration
    for (k in 1:N) {
      #cat("iteration:", k)
      xk = x
      yk = y
      thetak = theta
      # ck = 0.49

      ck = c*(k + 1)^p

      theta = prox_eta(x, y, theta)

      dkx = (1/ck) * F_x(x, y) + f_x(x, y) + g_x(x, y) - f_x(x, theta) - g_x(x, theta)
      x = pmax(x - alpha * dkx,0*e0)

      dky = (1/ck) * F_y(x, y) + f_y(x, y) - (y - theta)/gamma
      y = prox_beta(x, y, dky)

      res1[k] = norm(x - xk , "2") / norm(xk, "2")
      res2[k] = norm(y - yk, "2") / norm(yk, "2")
      res3[k] = norm(theta - thetak, "2") / norm(thetak, "2")
      arrF[k] = up_fun(x, y)
    }

    return(list(x = x, y = y, theta = theta, Xconv = res1, Yconv = res2, Thetaconv= res3, Fseq = arrF))
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

    result <- main_fun(A_val, b_val, A_tr, b_tr, group,N, alpha = alpha.seq[1], beta = beta.seq[1], eta = eta.seq[1], c = c, gamma = gamma, p = p)
    value[1] <- result$Fseq[order(result$Fseq, decreasing = FALSE)[1]]



    set.seed(123)
    for (j in 2:iter) {
      #T <- T*exp(-0.01*j)
      alpha.seq[j] <- rtruncnorm(n = 1, a = 0, mean = alpha.seq[j-1], sd = 1e-3)
      beta.seq[j] <- rtruncnorm(n = 1, a = 0, mean = beta.seq[j-1], sd = 1e-6)
      eta.seq[j] <- rtruncnorm(n = 1, a = 0, mean = eta.seq[j-1], sd = 1e-6)
      result <- main_fun(A_val, b_val, A_tr, b_tr, group,N, alpha = alpha.seq[j], beta = beta.seq[j], eta = eta.seq[j], c = c, gamma = gamma, p = p)
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

    return(main_fun(A_val, b_val, A_tr, b_tr, group, N, alpha = alpha.seq[opt_index], beta = beta.seq[opt_index], eta = eta.seq[opt_index], gamma, c, p))

  }

  else{
    return(main_fun(A_val, b_val, A_tr, b_tr, group, N, alpha, beta, eta, gamma, c, p))

  }


}
