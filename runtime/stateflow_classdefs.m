% mStateflow REPL surface — Tier 4f.
%
% Provides the `stateChart` classdef: a thin wrapper over the
% `<chart>_init` / `<chart>_tick` functions emitted by the chart
% lowering. Lets the user drive a chart from the REPL the same way
% they'd drive a `tf` / `pid` / OptimizationProblem instance.
%
% Usage from a REPL session, after running `matlabc -emit-matlab
% mychart.mflow > mychart.m` and source-include:
%
%   c = stateChart('mychart');
%   c = c.tick(struct('temp', 21), struct('tick', true));
%   regions = c.active();   % struct: region_id → active substate
%   c = c.emit('tick');     % broadcast an event into c.state
%   c = c.save_op('warm');
%   c = c.tick(struct('temp', 24), struct('tick', true));
%   c = c.restore_op('warm');
%
% Implementation note: handle-style (`obj = obj.method(...)`) so the
% chart's state mutates through value semantics. Same shape used by
% `tf` / `optimproblem` / `RFRational` in the rest of matlabc.

classdef stateChart
  properties
    name        % the originating chart name (the `Chart.Name`)
    state       % live chart state struct ({locals, regions, events, ...})
    init_fn     % function handle to <name>_init
    tick_fn     % function handle to <name>_tick
  end
  methods
    function obj = stateChart(name)
      % Looks up <name>_init / <name>_tick in scope. Before calling
      % this from the REPL, run:
      %   matlabc -emit-matlab my_chart.mflow > my_chart.m
      % and include `my_chart.m` in the workspace so the generated
      % `<name>_init` / `<name>_tick` symbols resolve. The Tier-N+
      % auto-load workflow (loadStateChart('foo.mflow')) lives in the
      % matlabc REPL prelude — out of scope for the classdef itself.
      obj.name = name;
      obj.init_fn = str2func([name '_init']);
      obj.tick_fn = str2func([name '_tick']);
      obj.state   = obj.init_fn();
    end

    function obj = tick(obj, inputs, events)
      if nargin < 2, inputs = struct(); end
      if nargin < 3, events = struct(); end
      [~, obj.state] = obj.tick_fn(obj.state, inputs, events);
    end

    function [outputs, obj] = step(obj, inputs, events)
      if nargin < 2, inputs = struct(); end
      if nargin < 3, events = struct(); end
      [outputs, obj.state] = obj.tick_fn(obj.state, inputs, events);
    end

    function obj = emit(obj, name)
      obj.state = mstateflow_emit(obj.state, name);
    end

    function regions = active(obj)
      regions = mstateflow_active(obj.state);
    end

    function obj = save_op(obj, name)
      obj.state = mstateflow_save_op(obj.state, name);
    end

    function obj = restore_op(obj, name)
      obj.state = mstateflow_restore_op(obj.state, name);
    end

    function obj = reset(obj)
      obj.state = obj.init_fn();
    end
  end
end
