% Control System Toolbox stdlib classdefs.
%
% Auto-prepended by matlabc when the user input mentions any of
% `tf` / `ss` / `zpk` / `pid` / `frd` as a call target or assignment
% LHS — see `findCstPrelude` and `userMentionsCstClass` in
% `tools/matlabc/main.cpp`. The §3.1 surface from
% docs/control_toolbox_roadmap.md.
%
% Slice 1 surface: `tf(num, den)` constructor + property storage +
% read access via `obj.Numerator` / `obj.Denominator`. Operator
% overloads (`G + H`, `G * H`, `tf('s')`, scalar mixing) are a
% follow-on slice — the missing piece is Sema-level type
% inference for class properties: today Sema returns `any` for any
% `obj.Field` read which then defaults to scalar f64 in the rest
% of the type-flow. Lowering.cpp now overrides the dispatch to
% `matlab_obj_get_mat` for CST classes specifically (so the
% RUNTIME call returns a matrix ptr), but the Sema-inferred type
% on the SSA value is still f64, which makes downstream
% expressions like `-obj.Numerator` lower as scalar arithmetic —
% the wrong shape. Closing that loop needs Sema to learn about
% CST property types (probably from constructor-body store
% analysis). Tracked in docs/control_toolbox_roadmap.md §12.

classdef tf
    properties
        Numerator
        Denominator
    end
    methods
        function obj = tf(num, den)
            if nargin == 2
                obj.Numerator = num;
                obj.Denominator = den;
            end
        end
    end
end
