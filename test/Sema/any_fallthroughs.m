% any_fallthroughs.m — pins the typing contract at the documented
% `return TC.any()` sites in TypeInference.cpp (#191 P1.2). Each result below
% is intentionally `[any]` today; a later precision item (P1.1 method results,
% P2.1 inter-procedural, P4.2 struct fields, P4.3 cell elements) will tighten
% one of these, at which point this golden updates as a visible signal.

% Cell brace-index element type — untracked (P4.3).
c = {1, 2, 3};
ce = c{2};

% Forward-referenced user function: its output type isn't available at the
% call site (defined later in the TU) — inter-procedural inference is P2.1.
fr = laterfn(4);

% Cell-index call form.
cc = c{1};

function r = laterfn(a)
  r = a + 1;
end
