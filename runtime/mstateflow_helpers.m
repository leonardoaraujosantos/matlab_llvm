% mStateflow runtime helpers (Tier 4c).
%
% Auto-pulled into the REPL / -simulate prelude whenever the user
% references one of the names below. Pure MATLAB — every chart-tick
% the matlabc lowering emits operates on the same state struct
% shape ({locals, regions, events}), so these helpers compose
% naturally with hand-written driver scripts.

function state = mstateflow_emit(state, name)
  % Broadcast a chart event in the current super-step. Equivalent to
  % the lowering's auto-emitted `state.events.<NAME> = true;`. Useful
  % from a REPL session or a driver script that doesn't go through
  % the events-struct argument.
  state.events.(name) = true;
end

function state = mstateflow_save_op(state, name)
  % Save a named operating point (active-state vector + chart-local
  % data block). Snapshots are stored under `state.snapshots.(name)`
  % and are themselves stripped of the `snapshots` field so the
  % storage doesn't grow recursively. Mirrors Stateflow's `Operating
  % Point` API (User's Guide §18) and the mflowLink snapshot ring.
  if ~isfield(state, 'snapshots'), state.snapshots = struct(); end
  snap = state;
  snap.snapshots = struct();
  state.snapshots.(name) = snap;
end

function state = mstateflow_restore_op(state, name)
  % Restore a previously-saved operating point. Preserves the
  % current `snapshots` table so the user can switch back and forth.
  if ~isfield(state, 'snapshots') || ~isfield(state.snapshots, name)
    error('mstateflow:unknownSnapshot', ...
          'no snapshot named "%s"', name);
  end
  saved = state.snapshots.(name);
  saved.snapshots = state.snapshots;
  state = saved;
end

function ids = mstateflow_active(state)
  % Return a struct of region-id → active-substate-id pairs so a
  % driver script can pretty-print the current active configuration.
  ids = struct();
  if ~isfield(state, 'regions'), return; end
  fnames = fieldnames(state.regions);
  for k = 1:numel(fnames)
    ids.(fnames{k}) = state.regions.(fnames{k});
  end
end

function state = mstateflow_reset(state)
  % Force re-initialisation on the next `<chart>_tick` invocation.
  state.initialized = false;
end

function state = mstateflow_push_history(state, cap)
  % Tier 6 — push the current state onto a ring of past super-step
  % snapshots, evicting the oldest when the cap is hit. Pairs with
  % `mstateflow_pop_history` to support step-back in driver scripts
  % the same way Stateflow's Operating Points API does. `state.history`
  % is a cell array; cap defaults to 256 to match the C-side ring.
  if nargin < 2, cap = 256; end
  if ~isfield(state, 'history'), state.history = {}; end
  snap = state;
  snap.history = {};
  state.history{end+1} = snap;
  if numel(state.history) > cap
    state.history = state.history(end-cap+1:end);
  end
end

function state = mstateflow_pop_history(state)
  % Step back one super-step boundary. Errors when the ring is empty.
  if ~isfield(state, 'history') || isempty(state.history)
    error('mstateflow:emptyHistory', 'no prior snapshot to step back to');
  end
  saved = state.history{end};
  saved.history = state.history(1:end-1);
  state = saved;
end

function state = mstateflow_auto_snap(state, cap)
  % Push the current state onto an auto-snapshot ring at the end of
  % each super-step. Capacity bounded so long runs don't grow without
  % limit; oldest entry is evicted when full. Pairs with the C-side
  % mstateflow_snapshot_* ring for cross-process step-back. The
  % lowering emits a call to this at the end of <chart>_tick when
  % state.auto_snapshot is true (off by default).
  if nargin < 2, cap = 32; end
  if ~isfield(state, 'auto_snaps'), state.auto_snaps = {}; end
  if ~isfield(state, 'auto_snapshot') || ~state.auto_snapshot
    return
  end
  snap = state;
  snap.auto_snaps = {};
  state.auto_snaps{end+1} = snap;
  if numel(state.auto_snaps) > cap
    state.auto_snaps = state.auto_snaps(end-cap+1:end);
  end
end
