"""The chat layer and task executor on the real SO-101 (issue #5 / #7 rules unchanged).

The executor (agentic_sim.task.executor.TaskExecutor) decides: orders, ledger, questions, shortage, no_return. On the
real arm its per-frame simulator skills are replaced by blocking routines that already work on the hardware (fixed
park <-> start route, D455 detection in the mount frame, program grasp, programmed place into the beige slots
P1-P4), run in a worker thread while the executor keeps stepping:

  tracker.Tracker       stable block IDs over D455 scans (the executor indexes blocks by ID)
  arm.RealArm / FakeArm  scan / pick / place / home on the follower (or on a simulated table for tests)
  skills                RealPick / RealPlace / RealHome / RealHold / RealObserve with the executor's Skill interface
  executor.RealTaskExecutor   TaskExecutor with those skills, D455 tray membership, P1-P4 and the safe grasp zone
  session.RealSession / RealHandler   the task_* protocol (tools/real_task_server.py), same as the simulators
"""
