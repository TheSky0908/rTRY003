#' Support Vector Machine Based on LVHBA
#' @description
#' A short description...
#'
#' @param features
#' @param labels
#' @param N Total iterations. Default is 200.
#' @param eta
#' @param alpha Proximal gradient stepsize of \code{x}. Default is 0.002.
#' @param beta Proximal gradient stepsize of \code{a} and \code{b}. Default is 0.002.
#' @param gamma1
#' @param gamma2
#' @param c_0
#' @param r
#' @param ub
#' @param lb
#' @param auto_tuning Whether an auto-hyperparameter-tuning is needed.Default is
#'     \code{FALSE}.
#' @param temperature Temperature of simulating annealing method for auto-
#'     hyperparameter-tuning. Default is 0.1.
#'
#' @return
#'
#'    \item{x}{}
#'    \item{y}{}
#'    \item{theta}{}
#'    \item{Xconv}{}
#'    \item{Yconv}{}
#'    \item{Thetaconv}{}
#'    \item{Fseq}{}
#'    \item{test_error_rate_seq}{}
#'    \item{Fseq_test}{}
#'    \item{val_error}{}
#'    \item{test_error}{}
#'    \item{test_error_rate}{}
#'
#'
#' @export
#'

LVHBA_SVM <- function(features, # 数据集特征
                      labels, # 数据集标签
                      T = 3, # T-fold交叉验证
                      N = 200, # 迭代步数
                      eta = 0.003, # 超参
                      alpha = 0.002, # 超参
                      beta = 0.002, # 超参
                      gamma1 = 1, # 超参
                      gamma2 = 1, # 超参
                      c_0 = 2, # 超参
                      r = 10, # 超参
                      ub = 10,# w_ub
                      lb = 10^(-6),#w_lb
                      auto_tuning = FALSE,
                      temperature = 0.1
){

  library(progress)
  library(truncnorm)

  # 用于求解投影步的package
  library(quadprog)
  # 用于拼接各个子问题的下层约束，得到便于电脑计算的矩阵
  library(Matrix)

  main_fun = function(features, # 数据集特征
                      labels, # 数据集标签
                      T, # T-fold交叉验证
                      N, # 迭代步数
                      eta, # 超参
                      alpha, # 超参
                      beta, # 超参
                      gamma1, # 超参
                      gamma2, # 超参
                      c_0, # 超参
                      r, # 超参
                      ub,# w_ub
                      lb#w_lb
  ){

    # 检查样本数与标签数是否匹配
    if (ncol(features) != ncol(labels)) {
      stop('the number of samples and labels should be iedntity!')
    }

    set.seed(15)  # 设置随机种子以确保结果可重现

    # 根据样本个数分割交叉验证集/ 测试集
    num_cols = ncol(features)
    num_cols_part1 = floor(num_cols / 2)  # 用于交叉验证的集合样本个数，等分
    num_cols_part2 = num_cols - num_cols_part1  # 测试集样本个数
    #cat("交叉验证集样本个数：", num_cols_part1, "\n")
    #cat("测试集样本个数：", num_cols_part2, "\n")

    # 随机选择要放入第一部分和第二部分的列索引，首先打乱1-num_cols(列数)的排列顺序
    col_indices = sample(1:num_cols, num_cols)
    # 将列索引分成两部分
    col_indices_part1 = col_indices[1:num_cols_part1] # 按照打乱后的顺序选择索引
    col_indices_part2 = col_indices[(num_cols_part1 + 1):num_cols]

    # 根据列索引选择特征矩阵的列（特征向量）
    A = as.matrix(features[, col_indices_part1])
    b = t(as.matrix(labels[, col_indices_part1]))
    A_test = as.matrix(features[, col_indices_part2])
    b_test = t(as.matrix(labels[, col_indices_part2]))
    # A = features
    # b = labels
    # A_test <- as.matrix(features[, col_indices_part2])
    # b_test <- t(as.matrix(labels[, col_indices_part2]))


    # 样本特征向量的总索引集
    data_index = 1:dim(A)[2]
    # 特征向量的维数
    p = dim(A)[1]
    #cat("样本向量维数p：", p, "\n")

    # 预先设置的箱约束
    wub = ub * matrix(1, p, 1)
    wlb = lb * matrix(1, p, 1)

    # 复制索引集，方便交叉验证分组
    data_index_copy = data_index
    # T-fold 验证下，计算每个验证集的大小
    tr_set_size = length(data_index) %/% T
    if (tr_set_size < 1) {
      stop('the data can\'T be divided into T groups\n')
    }

    # 创建一个空列表来存储train/validation set的样本索引集
    Omega_tr = list()
    Omega_val = list()

    # 将数据随机分割数据集成T个子集
    for (i in 1:(T - 1)) {
      tr_set_indices = sample(length(data_index), tr_set_size, replace = FALSE)
      # 无放回的抽样，选取第i个训练集的索引集
      Omega_tr[[i]] = data_index[tr_set_indices] # 第i个训练集样本索引j集
      Omega_val[[i]] = setdiff(data_index_copy, Omega_tr[[i]])
      # 将其在整个索引集中的补集作为第i个验证集样本索引集
      data_index = data_index[-tr_set_indices] # 剩下的样本索引集
    }

    # 将剩余的索引集作为第T个训练集样本索引集
    Omega_tr[[T]] = data_index
    Omega_val[[T]] = setdiff(data_index_copy, Omega_tr[[T]])

    # 用于记录每个样本集合的个数
    Omega_tr_size = matrix(rep(1), T)
    Omega_val_size = matrix(rep(1), T)

    # 定义一些初始值以及需要用到的矩阵
    lambda = 10 # 上层变量迭代初始值
    w_bar = 10^(-6) * matrix(rep(1), nrow = p ) # 上层变量迭代初始值
    c = matrix(rep(1), T) # 下层变量迭代初始值
    w = list() # 下层变量初始化空列表
    xi = list() # 下层变量初始化空列表
    E_xi = list() # 每一个\xi向量对应维度的单位矩阵空列表
    # e_xi = list() # 每一个\xi向量对应维度的全1向量空列表
    E_p = diag(1, p, p) # p维单位矩阵
    ones_p = matrix(1, p, 1)
    for (t in 1:T) {
      w[[t]] = matrix(0,nrow = p, ncol = 1 ) # 初始化每一个w^t
      Omega_tr_size[t] = length(Omega_tr[[t]]) # 获取tr set的大小
      Omega_val_size[t] = length(Omega_val[[t]]) # 获取val set的大小
      xi[[t]] = matrix(0,nrow = Omega_tr_size[t], ncol = 1) #初始化每一个\xi^t
      E_xi[[t]] = diag(1, Omega_tr_size[t], Omega_tr_size[t]) # 列表赋值
      # e_xi[[t]] = matrix(rep(0),nrow = Omega_tr_size[t]) # 列表赋值
    }

    # 上、下层变量统一初始值，x = (lambda, w_bar), y = (w^t, c^t, \xi^t),t = 1,...,T
    x = rbind(lambda,w_bar)
    y = matrix(1, nrow = T + p * T + ncol(A), ncol = 1)
    theta = matrix(1, nrow = T + p * T + ncol(A), ncol = 1) # 跟y规模相同的向量
    # 按变量顺序填写正确的y
    index_y = 0
    for (t in 1:T) {
      # 填充 w[[t]] 到 y 矩阵中的相应位置
      y[(index_y + 1):(index_y + p),] = w[[t]]
      # 填充 c[t] 到 y 矩阵中的相应位置
      y[index_y + p + 1,] = c[t]
      # 填充 xi[[t]] 到 y 矩阵中的相应位置
      y[(index_y + p + 1 + 1):(index_y + p + 1 + Omega_tr_size[t]),] = xi[[t]]
      # 更新 index_y 为下一个位置
      index_y = index_y + p + 1 + Omega_tr_size[t]
    }
    #cat("上层变量维数：", dim(x)[1], "\n")
    #cat("下层变量维数：", dim(y)[1], "\n")
    ## 整理下层约束
    # 定义S，下层约束系数矩阵
    S_x2 = matrix(0, nrow = 0, ncol = p) # \bar{w}对应的系数矩阵
    S_y = matrix(0, nrow = 0, ncol = 0)  # 下层变量y对应的系数矩阵
    # L， 下层约束的bound
    L = rbind(0, wub, -wlb)
    L_g = 0 * rbind(0, wub, -wlb)
    for (t in 1:T) {
      B_t = matrix(0, nrow = 0, ncol = p) # 初始化
      b_t = matrix(0, nrow = 0, ncol = 1) # 初始化
      for (j in Omega_tr[[t]]) {
        B_t = rbind(B_t, -b[j] * t(A[,j])) # 按行方向拼，直到拼接出完整的B_t
        b_t = rbind(b_t, b[j]) # 按行方向拼，直到拼接出完整的b_t
      }
      # 拼接出A_t
      A_t1 = cbind(-E_p, matrix(0, nrow = p, ncol = 1), matrix(0, nrow = p, ncol = ncol(E_xi[[t]]))) # A_t第1行块
      A_t2 = -A_t1 # A_t第2行块
      A_t3 = cbind(B_t, b_t, -E_xi[[t]]) # A_t第3行块
      A_t4 = cbind(matrix(0, nrow = nrow(E_xi[[t]]), ncol = p), matrix(0, nrow = nrow(E_xi[[t]]), ncol = 1), -E_xi[[t]])
      # A_t第4行块
      A_t = rbind(A_t1, A_t2, A_t3, A_t4) # 拼接A_t
      S_y = bdiag(S_y, A_t) # 拼接S_y, 下层变量y的系数矩阵
      M_t = rbind(-E_p, -E_p, matrix(0, nrow = nrow(E_xi[[t]]), ncol = p), matrix(0, nrow = nrow(E_xi[[t]]), ncol = p))
      S_x2 = rbind(S_x2, M_t) # 拼接S_x2, 上层变量w_bar对应的系数矩阵
      L = rbind(L, matrix(0, nrow = p, ncol = 1), matrix(0, nrow = p, ncol = 1), matrix(-1, nrow = nrow(E_xi[[t]]), ncol = 1), matrix(0, nrow = nrow(E_xi[[t]]), ncol = 1))
      L_g = rbind(L_g, matrix(0, nrow = p, ncol = 1), matrix(0, nrow = p, ncol = 1), matrix(-1, nrow = nrow(E_xi[[t]]), ncol = 1), matrix(0, nrow = nrow(E_xi[[t]]), ncol = 1))
    }


    S_x1 = matrix(-1, nrow = 1, ncol = 1) # S左上角子矩阵S_x1
    # 拼接S右下角子矩阵S_x2y, (w_bar,y)对应的系数矩阵
    S_x2y_12 = cbind( rbind(E_p, -E_p), matrix(0, 2 * p, T + p * T + ncol(A))) # S_x2y 第1，2行块

    S = as.matrix( bdiag(S_x1, rbind(S_x2y_12, cbind(S_x2, S_y) )) )

    S_g = as.matrix( bdiag(0 * S_x1, rbind(0 * S_x2y_12, cbind(S_x2, S_y) )) )
    #S = matrix(rep(0), nrow = 1 + 2 * p + 2 * p * T + ncol(A) * 2, ncol = 1 + p + T + p * T + ncol(A))
    #L = matrix(rep(1), nrow = nrow(S))*r

    # closed set Z
    ################################################################
    U = matrix(rep(1), nrow = nrow(S))*r
    z = matrix(rep(1), nrow = nrow(S))
    lam = matrix(rep(1), nrow = nrow(S)) # lambda in LV-HVA
    # theta 初始值
    theta_w = w
    theta_c = c
    theta_xi = xi


    # 近似目标函数
    up_fun = function(lambda, w_bar, w, c, xi){
      result = 0
      for (t in 1:T) {
        for (j in Omega_val[[t]]) {
          result = result + (1/Omega_val_size[t]) * log( 1 + exp(1 - b[j] * (t(A[, j]) %*% w[[t]] - c[t]) ))
        }
      }
      return((1/T)*result)
    }
    # 真实目标函数
    up_fun_real = function(lambda,w_bar, w, c, xi){
      result = 0
      for (t in 1:T) {
        for (j in Omega_val[[t]]) {
          result = result + (1/Omega_val_size[t]) * max(1 - b[j] * (t(A[, j]) %*% w[[t]] - c[t]),0)
        }
      }
      return((1/T)*result)
    }

    # 测试误差函数
    test_error_fun = function(lambda,w_bar, w, c, xi){
      result = 0
      for (t in 1:T) {
        for (j in 1:ncol(A_test)) {
          result = result + (1/ncol(A_test)) * max(1 - b_test[j] * (t(A_test[, j]) %*% w[[t]] - c[t]),0)
        }
      }
      return((1/T)*result)
    }


    # 测试误差率
    test_error_rate_fun = function(lambda,w_bar, w, c, xi){
      result = 0
      # 记录分类错误的个数
      for (j in 1:ncol(A_test)) {
        if (sign(t(A_test[, j]) %*% w[[t]] - c[t]) != sign(b_test[j])) {
          result = result + 1  # 分类错误则+1
        }
        # print(result)
      }
      # 得到分类错误率
      return((1/ncol(A_test))*result)
    }


    # # 测试误差率
    # test_error_rate_fun = function(lambda,w_bar, w, c, xi){
    #   result = 0
    #   # 记录分类错误的个数
    #   for (j in 1:ncol(A_test)) {
    #      result = result +  max(sign(1 - b_test[j] * (t(A_test[, j]) %*% w[[t]] - c[t])),0)
    #      print(result)
    #   }
    #   # 得到分类错误率
    #   return((1/ncol(A_test))*result)
    # }


    # 下层目标函数
    low_fun = function(lambda, w_bar, w, c, xi){
      result = 0
      for (t in 1:T) {
        result = result + 0.5 * lambda * norm(w[[t]],type = 2)^2 + sum(xi[t])
      }
    }

    #
    g = function(lambda, w_bar, w, c, xi){
      x = rbind(lambda, w_bar)
      y = matrix(0, nrow = 0, ncol = 1)
      for (t in 1:T) {
        y = rbind(y, as.matrix(w[[t]]), c[t], as.matrix(xi[[t]]))
      }
      x_y = rbind(x, y)
      return(S_g %*% x_y - L_g)
    }


    # 偏导
    F_x = function(lambda, w_bar, w, c, xi){
      result = matrix(0, p + 1, 1)
      return(result)
    }

    F_y <- function(lambda, w_bar, w, c, xi) {
      # Initialize gradients for w and c
      grad_w = list()
      grad_c = matrix(0, T, 1)
      grad_xi = list()
      result = matrix(0, 0, 1) # 初始化结果空向量

      # Calculate the partial derivatives
      for (t in 1:T) {
        grad_w[[t]] = 0 * ones_p
        # grad_c[t] = matrix(0,1,1)
        for (j in Omega_val[[t]]) {
          exp_term = exp(1 - b[j] * (t(A[,j]) %*% w[[t]] - c[t]))
          common_term = exp_term / (1 + exp_term)
          grad_w[[t]] = grad_w[[t]] - (1 / T) * (1 / length(Omega_val[[t]])) * as.numeric(common_term * b[j]) * A[,j]
          grad_c[t] = grad_c[t] + (1 / T) * (1 / length(Omega_val[[t]])) * common_term * b[j]
        }
        grad_xi[[t]] = matrix(rep(0),nrow = Omega_tr_size[t])
        result = rbind(result, grad_w[[t]], grad_c[t], grad_xi[[t]])
      }
      return(result)
    }

    f_x = function(lambda, w_bar, w, c, xi){
      result = matrix(0, p + 1, 1)
      return(result)
    }

    f_y = function(lambda, w_bar, w, c, xi){
      grad_w = list()
      grad_c = matrix(0, T, 1)
      grad_y = matrix(0, 0, 1)
      grad_xi = list()
      for (t in 1:T) {
        grad_w[[t]] = lambda * w[[t]]
        grad_c[t] = 0
        grad_xi[[t]] = matrix(1,nrow = Omega_tr_size[t], ncol = 1)
        grad_y = rbind(grad_y, as.matrix(grad_w[[t]]), grad_c[t],grad_xi[[t]])
      }
      return(list(grad_w = grad_w, grad_c = grad_c, grad_xi = grad_xi, grad_y = grad_y))
    }


    g_x = function(lambda, w_bar, theta_w, theta_c, theta_xi){
      result = S_g[, 1:(p + 1)]
      return(t(result))
    }

    g_y = function(lambda, w_bar, theta_w, theta_c, theta_xi){
      result = S_g[, (p + 2):ncol(S_g)]
      return(t(result))
    }


    up_fun_seq = numeric(N + 1) # 记录上层目标变化
    test_error_seq = numeric(N + 1) # 记录测试误差变化
    test_error_rate_seq = numeric(N + 1) #  记录测试误差率变化
    X_seq = numeric(N) # 记录上层变量变化
    Y_seq = numeric(N) # 记录下层变量变化
    Theta_seq = numeric(N) # 记录辅助变量变化
    up_fun_seq[1] = up_fun_real(lambda, w_bar, w, c, xi)
    test_error_seq[1] = test_error_fun(lambda, w_bar, w, c, xi)
    test_error_rate_seq[1] = 100 * test_error_rate_fun(lambda, w_bar, w, c, xi)


    # 迭代求解
    for (k in 1:N) {
      xk = x
      yk = y
      thetak = theta
      ck = c_0 * (k + 1)^0.48
      grad_f_y = f_y(lambda, w_bar, theta_w, theta_c, theta_xi)
      dk_theta = grad_f_y$grad_y + g_y(lambda, w_bar, theta_w, theta_c, theta_xi) %*% lam + (theta - y) / gamma1
      dk_lam = -g(lambda, w_bar, theta_w, theta_c, theta_xi) + (lam - z) / gamma2

      theta = theta - (eta * dk_theta)
      theta_w <- list()
      theta_c <- matrix(rep(0), nrow = T)
      theta_xi = list()
      # 更新 theta
      index_theta = 0
      for (t in 1:T) {
        theta_w[[t]] = theta[(index_theta + 1):(index_theta + p), , drop = FALSE] # drop, 保留矩阵二维结构
        theta_c[t] = theta[index_theta + p + 1, , drop = FALSE]
        theta_xi[[t]] = theta[(index_theta + p + 1 + 1):(index_theta + p + 1 + Omega_tr_size[t]), , drop = FALSE]
        index_theta = index_theta + p + 1 + Omega_tr_size[t]
        # print(index_theta)
      }


      lam = lam - (eta * dk_lam)
      lam = pmin(pmax(lam, 0), U)

      dkx = F_x(lambda, w_bar, w, c, xi) / ck + f_x(lambda, w_bar, w, c, xi) - f_x(lambda, w_bar, theta_w, theta_c, theta_xi) - g_x(lambda, w_bar, theta_w, theta_c, theta_xi) %*% lam
      dky = F_y(lambda, w_bar, w, c, xi) / ck + f_y(lambda, w_bar, w, c, xi)$grad_y - (y - theta) / gamma1
      x = x - alpha * dkx
      y = y - alpha * dky
      x_y = rbind(x, y)

      # 求解投影
      x_y = as.matrix(solve.QP(Dmat = diag(1, ncol(S), ncol(S)), dvec =  x_y, Amat = t(-S), bvec = -L)$solution)

      x = x_y[1:(1 + p), , drop = FALSE]
      # 更新下层变量 lambda，w_bar
      lambda = x[1]
      w_bar = x[2:(p + 1), , drop = FALSE]

      y = x_y[(2 + p):length(x_y), ,drop = FALSE]

      #######################
      # index = 1
      # for (t in 1:T) {
      #   w[[t]] <- y[index:(index + p - 1),,drop = FALSE]
      #   c[t] <- y[index + p,]
      #   xi[[t]] = y[(index + p + 1):(index + p + Omega_tr_size[t]),,drop = FALSE]
      # }
      index_y = 0
      for (t in 1:T) {
        w[[t]] = y[(index_y + 1):(index_y + p), , drop = FALSE] # drop, 保留矩阵二维结构
        c[t] = y[index_y + p + 1, , drop = FALSE]
        xi[[t]] = y[(index_y + p + 1 + 1):(index_y + p + 1 + Omega_tr_size[t]), , drop = FALSE]
        index_y = index_y + p + 1 + Omega_tr_size[t]
      }


      # for (t in 1:T) {
      #   w[[t]] <- y[t:(t + p - 1),,drop = FALSE]
      #   c[t] <- y[t + p,]
      #   xi[[t]] = y[(t + p + 1):(t + p + Omega_tr_size[t]),,drop = FALSE]
      # }

      dkz = -(lam - z) / gamma2
      z = z - (beta * dkz)
      z = pmin(pmax(z, 0),U)

      X_seq[k] = norm(x - xk , "2") / norm(xk, "2")
      Y_seq[k] = norm(y - yk, "2") / norm(yk, "2")
      Theta_seq[k] = norm(theta - thetak, "2") / norm(thetak, "2")
      up_fun_seq[k + 1] = up_fun_real(lambda, w_bar, w, c, xi)
      test_error_seq[k + 1] = test_error_fun(lambda, w_bar, w, c, xi)
      test_error_rate_seq[k + 1] = 100 * test_error_rate_fun(lambda, w_bar, w, c, xi)
    }

    val_error = up_fun_real(lambda, w_bar, w, c, xi)
    test_error = test_error_fun(lambda, w_bar, w, c, xi)
    test_error_rate = 100 * test_error_rate_fun(lambda, w_bar, w, c, xi)
    #print(val_error)
    #print(test_error)
    #print(test_error_rate)
    return(list(x = x, y = y, theta = theta, Xconv = X_seq, Yconv = Y_seq,
                Thetaconv = Theta_seq, Fseq = up_fun_seq, test_error_rate_seq = test_error_rate_seq,
                Fseq_test = test_error_seq, val_error = val_error, test_error = test_error,
                test_error_rate = test_error_rate))
  }

  if(auto_tuning == TRUE){
    message("\n","Auto-hyperparameters-tuning is proceeding now.")

    iter <- 100

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

    result <- main_fun(features, # 数据集特征
                       labels, # 数据集标签
                       T, # T-fold交叉验证
                       N, # 迭代步数
                       eta = eta.seq[1], # 超参
                       alpha = alpha.seq[1], # 超参
                       beta = beta.seq[1], # 超参
                       gamma1 = gamma1, # 超参
                       gamma2 = gamma2, # 超参
                       c_0 = c_0, # 超参
                       r = r, # 超参
                       ub = ub,# w_ub
                       lb = lb #w_lb
    )

    value[1] <- result$val_error[order(result$val_error, decreasing = FALSE)[1]]

    set.seed(123)
    for (j in 2:iter) {
      #T <- T*exp(-0.01*j)
      alpha.seq[j] <- rtruncnorm(n = 1, a = 0, mean = alpha.seq[j-1], sd = 1e-3)
      beta.seq[j] <- rtruncnorm(n = 1, a = 0, mean = beta.seq[j-1], sd = 1e-6)
      eta.seq[j] <- rtruncnorm(n = 1, a = 0, mean = eta.seq[j-1], sd = 1e-6)
      result <-   main_fun(features, # 数据集特征
                           labels, # 数据集标签
                           T, # T-fold交叉验证
                           N, # 迭代步数
                           eta = eta.seq[j], # 超参
                           alpha = alpha.seq[j], # 超参
                           beta = beta.seq[j], # 超参
                           gamma1 = gamma1, # 超参
                           gamma2 = gamma2, # 超参
                           c_0 = c_0, # 超参
                           r = r, # 超参
                           ub = ub,# w_ub
                           lb = lb #w_lb
      )

      candidate <- result$val_error[order(result$val_error, decreasing = FALSE)[1]]




      if(candidate > value[j-1] & runif(n = 1) > exp((value[j-1]-candidate)/temperature)){
        value[j] <- value[j-1]
      } else {
        value[j] <- candidate
      }
      pb$tick()
    }


    opt_index <- order(value)[1]

    cat("\n", "Auto-hyperparameters-tuning is done.")
    cat("\nFinal hyper-paramaters (alpha,beta,eta) are chosen as:",c(alpha.seq[opt_index], beta.seq[opt_index], eta.seq[opt_index]))

    return(main_fun(features, # 数据集特征
                    labels, # 数据集标签
                    T, # T-fold交叉验证
                    N, # 迭代步数
                    eta = eta.seq[opt_index], # 超参
                    alpha = alpha.seq[opt_index], # 超参
                    beta = beta.seq[opt_index], # 超参
                    gamma1 = gamma1, # 超参
                    gamma2 = gamma2, # 超参
                    c_0 = c_0, # 超参
                    r = r, # 超参
                    ub = ub,# w_ub
                    lb = lb #w_lb
    ))

  }
  else{
    main_fun(features, # 数据集特征
             labels, # 数据集标签
             T, # T-fold交叉验证
             N, # 迭代步数
             eta, # 超参
             alpha, # 超参
             beta, # 超参
             gamma1 = gamma1, # 超参
             gamma2 = gamma2, # 超参
             c_0 = c_0, # 超参
             r = r, # 超参
             ub = ub,# w_ub
             lb = lb #w_lb
    )
  }



}
