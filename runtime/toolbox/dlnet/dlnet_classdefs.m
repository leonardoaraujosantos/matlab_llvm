% Deep Learning Toolbox — classdef umbrella (Tiers 1–2: dlarray + autodiff).
% Auto-prepended by matlabc when the user input mentions a deep-learning symbol
% (`dlarray`, `dlgradient`, `extractdata`, `relu`, `softmax`, ...).
%
% `dlarray` is a value carrier holding the data matrix + a tape-node id.  Its
% operator methods (`plus`/`mtimes`/...) are dispatched automatically by the
% compiler's classdef operator-overloading path; the activation/reduction
% methods (`relu`/`softmax`/`sum`/...) are dispatched by a CallOrIndex arm when
% the argument is a `dlarray`.  Each method allocates a fresh `dlarray` shell
% and forwards to a `matlab_dlnet_*` runtime entry that computes the forward
% value, records a reverse-mode tape node, and populates the shell.

classdef dlarray
    properties
        Data matrix      % value
        Id               % tape-node index (-1 = untracked constant)
    end
    methods
        function obj = dlarray()
            obj.Data = zeros(0, 0);
            obj.Id   = -1;
        end

        % ---- operator overloading (binary) --------------------------------
        function r = plus(a, b)
            r = dlarray();
            matlab_dlnet_plus(r, a, b);
        end
        function r = minus(a, b)
            r = dlarray();
            matlab_dlnet_minus(r, a, b);
        end
        function r = mtimes(a, b)
            r = dlarray();
            matlab_dlnet_mtimes(r, a, b);
        end
        function r = times(a, b)
            r = dlarray();
            matlab_dlnet_times(r, a, b);
        end

        % ---- activations + reductions (function-call dispatch) ------------
        function r = relu(x)
            r = dlarray();
            matlab_dlnet_relu(r, x);
        end
        function r = sigmoid(x)
            r = dlarray();
            matlab_dlnet_sigmoid(r, x);
        end
        function r = tanh(x)
            r = dlarray();
            matlab_dlnet_tanh(r, x);
        end
        function r = softmax(x)
            r = dlarray();
            matlab_dlnet_softmax(r, x);
        end
        function r = sum(x)
            r = dlarray();
            matlab_dlnet_sum(r, x);
        end
        function r = mean(x)
            r = dlarray();
            matlab_dlnet_mean(r, x);
        end
        function r = log(x)
            r = dlarray();
            matlab_dlnet_log(r, x);
        end
        function r = exp(x)
            r = dlarray();
            matlab_dlnet_exp(r, x);
        end
        function r = transpose(x)
            r = dlarray();
            matlab_dlnet_transpose(r, x);
        end
        function r = embed(E, idx)
            r = dlarray();
            matlab_dlnet_embed(r, E, idx);
        end
        function r = ctranspose(x)
            r = dlarray();
            matlab_dlnet_transpose(r, x);
        end
        function r = crossentropy(y, t)
            r = dlarray();
            matlab_dlnet_crossentropy(r, y, t);
        end
        function r = mse(y, t)
            r = dlarray();
            matlab_dlnet_mse(r, y, t);
        end

        % ---- recurrent (functional LSTM, T4) ------------------------------
        function r = lstm(x, h0, c0, W, R, b)
            r = dlarray();
            matlab_dlnet_lstm(r, x, h0, c0, W, R, b);
        end
        function r = gru(x, h0, W, R, b)
            r = dlarray();
            matlab_dlnet_gru(r, x, h0, W, R, b);
        end
        function r = bilstm(x, h0f, c0f, h0b, c0b, W, R, b)
            r = dlarray();
            matlab_dlnet_bilstm(r, x, h0f, c0f, h0b, c0b, W, R, b);
        end
        function r = lstmp(x, h0, c0, W, R, P, b)
            r = dlarray();
            matlab_dlnet_lstmp(r, x, h0, c0, W, R, P, b);
        end

        % ---- Phase 1 small ops --------------------------------------------
        % Element-wise divide, sqrt, dim-aware mean, plus several modern
        % activation functions (leakyrelu / gelu / swish / softplus / elu).
        function r = rdivide(a, b)
            r = dlarray();
            matlab_dlnet_rdivide(r, a, b);
        end
        function r = sqrt(x)
            r = dlarray();
            matlab_dlnet_sqrt(r, x);
        end
        function r = mean_dim(x, dim)
            r = dlarray();
            matlab_dlnet_mean_dim(r, x, dim);
        end
        function r = leakyrelu(x)
            r = dlarray();
            matlab_dlnet_leakyrelu(r, x);
        end
        function r = gelu(x)
            r = dlarray();
            matlab_dlnet_gelu(r, x);
        end
        function r = swish(x)
            r = dlarray();
            matlab_dlnet_swish(r, x);
        end
        function r = softplus(x)
            r = dlarray();
            matlab_dlnet_softplus(r, x);
        end
        function r = elu(x)
            r = dlarray();
            matlab_dlnet_elu(r, x);
        end

        % ---- Tier C: rank-4 batched convolution (autodiff-tracked) ---------
        function r = conv2d_batch(x, w)
            r = dlarray();
            matlab_dlnet_conv2d_batch(r, x, w);
        end

        % reshape — method-by-arity split at the lowering arm picks
        % reshape2 (1-arg target) vs reshape4 (3-arg target).
        function r = reshape2(x, m, n)
            r = dlarray();
            matlab_dlnet_reshape2(r, x, m, n);
        end
        function r = reshape4(x, d1, d2, d3, d4)
            r = dlarray();
            matlab_dlnet_reshape4(r, x, d1, d2, d3, d4);
        end

        % Non-overlapping 2-D pooling (stride = kernel).
        function r = maxpool2d(x, kH, kW)
            r = dlarray();
            matlab_dlnet_maxpool2d(r, x, kH, kW);
        end
        function r = avgpool2d(x, kH, kW)
            r = dlarray();
            matlab_dlnet_avgpool2d(r, x, kH, kW);
        end

        % BatchNorm: per-channel normalize over (H, W, N) of a 4-D X,
        % scaled by γ (length C) + shifted by β (length C).
        function r = batchnorm(x, gamma, beta)
            r = dlarray();
            matlab_dlnet_batchnorm(r, x, gamma, beta);
        end

        % Full conv: optional bias + zero-padding + stride.
        %   conv2d_full(X, W, b, pad_h, pad_w, stride_h, stride_w)
        function r = conv2d_full(x, w, b, pad_h, pad_w, stride_h, stride_w)
            r = dlarray();
            matlab_dlnet_conv2d_full(r, x, w, b, pad_h, pad_w, stride_h, stride_w);
        end

        % softmax(X, dim) — softmax along a specified axis (matN-aware).
        function r = softmax_dim(x, dim)
            r = dlarray();
            matlab_dlnet_softmax_dim(r, x, dim);
        end
    end
end
% `extractdata(x)` and `dlgradient(loss, v)` are intercepted in Lowering.cpp
% (both return a plain numeric matrix — dlgradient yields the gradient of the
% scalar loss w.r.t. the dlarray `v`).
