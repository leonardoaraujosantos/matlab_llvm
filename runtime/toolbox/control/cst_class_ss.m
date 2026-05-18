% State-space model — auto-prepended when matlabc sees `ss(` or
% `ss =` in the user input. See cst_classdefs.m for the umbrella
% comment + the tf classdef.

classdef ss
    properties
        A
        B
        C
        D
    end
    methods
        function obj = ss(A, B, C, D)
            % `ss(A, B, C, D)` — state-space model:
            %   x' = A x + B u
            %   y  = C x + D u
            % All four are ordinary matrices (n×n, n×m, p×n, p×m).
            % Operator overloads:
            %   ss + ss → parallel: block-diagonal A, stacked B,
            %             concatenated C, summed D.
            %   ss * ss → series cascade: a*b means u → b → a → y;
            %             A = [A_a, B_a*C_b; 0, A_b], B = [B_a*D_b;
            %             B_b], C = [C_a, D_a*C_b], D = D_a*D_b.
            %   -ss     → negate output: C → -C, D → -D.
            % `feedback(ss, ss)` is exposed via the matrix-arg
            % primitive `feedback_ss` (CST roadmap Tier 1.5) — a
            % classdef-level `feedback` overload is a follow-on.
            if nargin >= 1, obj.A = A; end
            if nargin >= 2, obj.B = B; end
            if nargin >= 3, obj.C = C; end
            if nargin >= 4, obj.D = D; end
        end

        function r = plus(a, b)
            % Parallel: stack the two state spaces, sum their
            % outputs at u → (y_a + y_b).
            n1 = size(a.A, 1);
            n2 = size(b.A, 1);
            Z12 = zeros(n1, n2);
            Z21 = zeros(n2, n1);
            top   = horzcat(a.A, Z12);
            bot   = horzcat(Z21, b.A);
            new_A = vertcat(top, bot);
            new_B = vertcat(a.B, b.B);
            new_C = horzcat(a.C, b.C);
            new_D = a.D + b.D;
            r = ss(new_A, new_B, new_C, new_D);
        end

        function r = minus(a, b)
            % Parallel difference: y_a - y_b.
            n1 = size(a.A, 1);
            n2 = size(b.A, 1);
            Z12 = zeros(n1, n2);
            Z21 = zeros(n2, n1);
            top   = horzcat(a.A, Z12);
            bot   = horzcat(Z21, b.A);
            new_A = vertcat(top, bot);
            new_B = vertcat(a.B, b.B);
            new_C = horzcat(a.C, -b.C);
            new_D = a.D - b.D;
            r = ss(new_A, new_B, new_C, new_D);
        end

        function r = mtimes(a, b)
            % Series cascade: a * b ≡ u → b → a → y.
            % State [x_a; x_b] with x_b's output driving x_a.
            n1 = size(a.A, 1);
            n2 = size(b.A, 1);
            Z21 = zeros(n2, n1);
            top   = horzcat(a.A, a.B * b.C);
            bot   = horzcat(Z21, b.A);
            new_A = vertcat(top, bot);
            new_B = vertcat(a.B * b.D, b.B);
            new_C = horzcat(a.C, a.D * b.C);
            new_D = a.D * b.D;
            r = ss(new_A, new_B, new_C, new_D);
        end

        function r = uminus(a)
            r = ss(a.A, a.B, -a.C, -a.D);
        end
    end
end
