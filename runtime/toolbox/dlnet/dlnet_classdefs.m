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
    end
end
% `extractdata(x)` and `dlgradient(loss, v)` are intercepted in Lowering.cpp
% (both return a plain numeric matrix — dlgradient yields the gradient of the
% scalar loss w.r.t. the dlarray `v`).
