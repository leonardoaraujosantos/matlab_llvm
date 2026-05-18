/* Direct unit tests for the mStateflow runtime hooks:
 * DAP event queue (listening flag + drain count + reset), bounded
 * FIFO event queue (push / pop / count / capacity), and snapshot
 * ring (save_blob / copy / count / reset). */

#include "runtime_test.h"

/* Forward decls — runtime_mstateflow.cpp entries.
 * These use the `mstateflow_*` prefix (no `matlab_` namespace). */
void mstateflow_set_listening      (int On);
int  mstateflow_is_listening       (void);
void mstateflow_dap_state_enter    (const char *Id, double T);
void mstateflow_dap_state_exit     (const char *Id, double T);
void mstateflow_dap_transition_fired(const char *Id, const char *Src,
                                      const char *Dst, double T,
                                      const char *Event);
void mstateflow_dap_superstep_begin(double T, int Iter);
void mstateflow_dap_superstep_end  (double T, int Iter, int Quiescent);
void mstateflow_dap_event_broadcast(const char *Name, double T);
int  mstateflow_drain_count        (void);
void mstateflow_drain_reset        (void);
int  mstateflow_snapshot_save_blob (const char *Name, const void *Data,
                                     size_t Len);
int  mstateflow_snapshot_copy      (const char *Name, void *Out, size_t Cap);
int  mstateflow_snapshot_count     (void);
void mstateflow_snapshot_reset     (void);
void mstateflow_event_set_capacity (int Cap);
int  mstateflow_event_push         (const char *Name);
int  mstateflow_event_count        (void);
int  mstateflow_event_pop          (char *Out, size_t Cap);
void mstateflow_event_reset        (void);

/* ===== Listener flag ===== */

static void test_listening_toggle(void) {
    mstateflow_set_listening(1);
    RT_CHECK(mstateflow_is_listening() == 1, "set_listening(1) takes effect");
    mstateflow_set_listening(0);
    RT_CHECK(mstateflow_is_listening() == 0, "set_listening(0) clears");
}

/* ===== DAP event drain queue ===== */

static void test_drain_starts_empty(void) {
    mstateflow_set_listening(1);
    mstateflow_drain_reset();
    RT_CHECK(mstateflow_drain_count() == 0, "drain empty after reset");
    mstateflow_set_listening(0);
}

static void test_drain_records_state_enter(void) {
    mstateflow_set_listening(1);
    mstateflow_drain_reset();
    mstateflow_dap_state_enter("State_A", 1.0);
    mstateflow_dap_state_enter("State_B", 2.0);
    int n = mstateflow_drain_count();
    RT_CHECK(n == 2, "drain count == 2 after two enters");
    mstateflow_drain_reset();
    RT_CHECK(mstateflow_drain_count() == 0, "drain clears after reset");
    mstateflow_set_listening(0);
}

static void test_drain_ignored_when_not_listening(void) {
    mstateflow_drain_reset();
    mstateflow_set_listening(0);
    /* With listener off, events should be dropped silently. */
    mstateflow_dap_state_enter("State_A", 0.0);
    mstateflow_dap_state_exit("State_A", 0.5);
    mstateflow_dap_transition_fired("t1", "A", "B", 0.6, "evt");
    mstateflow_dap_superstep_begin(1.0, 0);
    mstateflow_dap_superstep_end(1.0, 0, 1);
    mstateflow_dap_event_broadcast("E", 1.5);
    RT_CHECK(mstateflow_drain_count() == 0,
             "events dropped without listener");
}

static void test_drain_event_broadcast(void) {
    mstateflow_set_listening(1);
    mstateflow_drain_reset();
    mstateflow_dap_event_broadcast("Trigger", 5.0);
    RT_CHECK(mstateflow_drain_count() == 1, "broadcast recorded");
    mstateflow_set_listening(0);
    mstateflow_drain_reset();
}

/* ===== Bounded FIFO event queue ===== */

static void test_fifo_push_pop_round_trip(void) {
    mstateflow_event_reset();
    mstateflow_event_set_capacity(8);
    int ok = mstateflow_event_push("HELLO");
    RT_CHECK(ok == 1, "push HELLO succeeds");
    RT_CHECK(mstateflow_event_count() == 1, "count after push");

    char buf[64] = {0};
    int popped = mstateflow_event_pop(buf, sizeof(buf));
    RT_CHECK(popped == 1, "pop returns 1 (success)");
    RT_CHECK(strcmp(buf, "HELLO") == 0, "pop returns same string");
    RT_CHECK(mstateflow_event_count() == 0, "queue empty after pop");
}

static void test_fifo_fifo_order(void) {
    mstateflow_event_reset();
    mstateflow_event_set_capacity(4);
    mstateflow_event_push("FIRST");
    mstateflow_event_push("SECOND");
    mstateflow_event_push("THIRD");
    RT_CHECK(mstateflow_event_count() == 3, "3 events queued");

    char buf[64] = {0};
    mstateflow_event_pop(buf, sizeof(buf));
    RT_CHECK(strcmp(buf, "FIRST") == 0, "FIFO: FIRST out first");
    mstateflow_event_pop(buf, sizeof(buf));
    RT_CHECK(strcmp(buf, "SECOND") == 0, "FIFO: SECOND next");
    mstateflow_event_pop(buf, sizeof(buf));
    RT_CHECK(strcmp(buf, "THIRD") == 0, "FIFO: THIRD last");
}

static void test_fifo_capacity_bounds(void) {
    /* The runtime FIFO is bounded with drop-oldest semantics, not
     * reject-on-full — push always returns 1 and the oldest entry
     * is evicted when capacity is exceeded. */
    mstateflow_event_reset();
    mstateflow_event_set_capacity(2);
    int a = mstateflow_event_push("A");
    int b = mstateflow_event_push("B");
    int c = mstateflow_event_push("C");
    RT_CHECK(a == 1 && b == 1 && c == 1, "all pushes accepted");
    RT_CHECK(mstateflow_event_count() == 2, "count clamps at capacity");
    /* The remaining entries should be B + C (A evicted). */
    char buf[64] = {0};
    mstateflow_event_pop(buf, sizeof(buf));
    RT_CHECK(strcmp(buf, "B") == 0, "drop-oldest: B is next");
    mstateflow_event_pop(buf, sizeof(buf));
    RT_CHECK(strcmp(buf, "C") == 0, "drop-oldest: C is last");
}

static void test_fifo_pop_empty_returns_zero(void) {
    mstateflow_event_reset();
    char buf[64] = {0};
    int popped = mstateflow_event_pop(buf, sizeof(buf));
    RT_CHECK(popped == 0, "pop empty returns 0");
}

/* ===== Snapshot ring ===== */

static void test_snapshot_save_and_copy(void) {
    mstateflow_snapshot_reset();
    const char payload[] = {1, 2, 3, 4, 5};
    int saved = mstateflow_snapshot_save_blob("snap1", payload, sizeof(payload));
    RT_CHECK(saved == 1, "snapshot save succeeded");
    RT_CHECK(mstateflow_snapshot_count() == 1, "snapshot count == 1");

    char out[16] = {0};
    int copied = mstateflow_snapshot_copy("snap1", out, sizeof(out));
    RT_CHECK(copied == (int)sizeof(payload), "snapshot copy returns length");
    for (size_t i = 0; i < sizeof(payload); ++i)
        RT_CHECK(out[i] == payload[i], "snapshot payload roundtrip");
}

static void test_snapshot_overwrite(void) {
    mstateflow_snapshot_reset();
    const char p1[] = {10, 20, 30};
    const char p2[] = {99};
    mstateflow_snapshot_save_blob("k", p1, sizeof(p1));
    mstateflow_snapshot_save_blob("k", p2, sizeof(p2));
    /* Overwriting the same name shouldn't double the count. */
    RT_CHECK(mstateflow_snapshot_count() == 1, "overwrite keeps count at 1");
    char out[16] = {0};
    int copied = mstateflow_snapshot_copy("k", out, sizeof(out));
    RT_CHECK(copied == 1, "second save replaces first");
    RT_CHECK(out[0] == 99, "new payload wins");
}

static void test_snapshot_missing_key(void) {
    mstateflow_snapshot_reset();
    char out[8] = {0};
    int copied = mstateflow_snapshot_copy("nonexistent", out, sizeof(out));
    RT_CHECK(copied == 0, "missing key returns 0");
}

static void test_snapshot_reset_clears_all(void) {
    mstateflow_snapshot_reset();
    const char p[] = {1};
    mstateflow_snapshot_save_blob("a", p, 1);
    mstateflow_snapshot_save_blob("b", p, 1);
    RT_CHECK(mstateflow_snapshot_count() == 2, "two snapshots saved");
    mstateflow_snapshot_reset();
    RT_CHECK(mstateflow_snapshot_count() == 0, "reset clears all");
}

int main(void) {
    RT_RUN(test_listening_toggle);
    RT_RUN(test_drain_starts_empty);
    RT_RUN(test_drain_records_state_enter);
    RT_RUN(test_drain_ignored_when_not_listening);
    RT_RUN(test_drain_event_broadcast);
    RT_RUN(test_fifo_push_pop_round_trip);
    RT_RUN(test_fifo_fifo_order);
    RT_RUN(test_fifo_capacity_bounds);
    RT_RUN(test_fifo_pop_empty_returns_zero);
    RT_RUN(test_snapshot_save_and_copy);
    RT_RUN(test_snapshot_overwrite);
    RT_RUN(test_snapshot_missing_key);
    RT_RUN(test_snapshot_reset_clears_all);
    RT_DONE();
}
