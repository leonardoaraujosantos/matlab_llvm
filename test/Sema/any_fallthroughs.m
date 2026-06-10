% any_fallthroughs.m — pins the typing contract at the documented
% `return TC.any()` sites in TypeInference.cpp (#191 P1.2). Each result below
% is intentionally `[any]`; a later precision item will tighten the remaining
% gaps (P1.1 method results, P2.1 inter-procedural, P4.2 struct fields). When
% one lands, this golden updates as a visible signal — as the cell brace-index
% case already did when P4.3 (cell element typing) shipped.

% Forward-referenced user function: its output type isn't available at the
% call site (defined later in the TU) — inter-procedural inference is P2.1.
fr = laterfn(4);

% A heterogeneous cell joins its element types to Any, so a brace-index is
% correctly untyped (the homogeneous case is precise — see cell_element_typing).
mixed = {1, "two"};
mx = mixed{1};

function r = laterfn(a)
  r = a + 1;
end
