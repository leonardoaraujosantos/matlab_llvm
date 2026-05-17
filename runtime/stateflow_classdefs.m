% mStateflow REPL surface — Tier 4f (rev. 2026-05-17).
%
% Provides the `stateChart` classdef: a thin invocation wrapper over
% the `<name>_tick` function the chart lowering emits.
%
% **Important shape note.** The persistent-scalar lowering (since
% 2026-05) does NOT use a state-struct argument. Each chart compiles
% to a single function
%
%   function [out_y1, out_y2, ...] = <name>_tick(in_x1, in_x2, ..., ev_e1, ev_e2, ...)
%
% with persistent state living inside the function (zeroed on first
% call via `isempty(init_done)`). That's why the previous classdef's
% `obj.state` / `obj.init_fn` design is gone — there is no state
% struct to thread through.
%
% Typical usage from a REPL session:
%
%   >> matlabc -emit-matlab mychart.mflow > mychart.m   % offline
%   >> % then either:
%   >> loadStateChart('mychart.mflow')                  % REPL builtin
%   >> [r, y, g] = traffic_light_tick(false);            % drive directly
%
% Or via the wrapper:
%
%   c = stateChart('traffic_light');
%   outs = c.tick({false});           % positional cell — events last
%   c.reset();                        % zero persistents
%
% For introspection / save-op / event-broadcast workflows that the
% old API exposed via `obj.emit` / `obj.active` / `obj.save_op` /
% `obj.restore_op`, drive the chart through the DAP server instead:
%
%   $ matlabc -simulate --sim-dap mychart.mflow
%
% and use the `stateChart/setLocal`, `stateChart/emit`,
% `stateChart/saveOperatingPoint`, etc. requests documented in
% docs/mStateflow_roadmap.md §6.7. The DAP path is the
% authoritative "live state" interface; the classdef here is a
% terse direct-invoke convenience.

classdef stateChart
  % Metadata-only handle on a lowered chart. matlabc's REPL JIT
  % doesn't currently lower `str2func` / `feval` calls, so the
  % classdef can't dynamically resolve the chart's tick function
  % from a name string. The pragmatic surface is therefore:
  %
  %   loadStateChart('foo.mflow');          % brings foo_tick into scope
  %   c = stateChart('foo');                % stores metadata
  %   [a, b, c_out] = foo_tick(in1, ev_e);  % drive directly
  %
  % The classdef stays as a discoverable type + a documentation
  % anchor; future work could either (a) generate a chart-specific
  % classdef per emitted .m, or (b) close the matlabc JIT gap on
  % function-handle / feval lowering.
  properties
    name        % the originating chart name
  end
  methods
    function obj = stateChart(name)
      obj.name = name;
    end
  end
end
