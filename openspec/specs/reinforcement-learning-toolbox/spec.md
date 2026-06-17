# Reinforcement Learning Toolbox Spec

## Purpose
Documents the shipped subset of the Reinforcement Learning Toolbox in the matlab_llvm compiler: a self-contained deep-RL stack built on the shipped Deep Learning autodiff engine, spanning tabular, value-based, policy-gradient, and continuous-control agents with predefined environments and a unified train/sim API. Shipped and merged via PR #83 (2026-05-31). (doc: docs/reinforcement_learning_toolbox_roadmap.md) (src: runtime/toolbox/rl)

## Requirements

### Requirement: Tabular and value-based agents
The system SHALL provide tabular and deep value-based agents. (src: runtime/toolbox/rl/rl_classdefs.m) (src: runtime/toolbox/rl/runtime_rl.cpp)

#### Scenario: Construct a value-based agent
- **WHEN** a program constructs `rlQAgent`, `rlSARSAAgent`, or `rlDQNAgent`
- **THEN** the system SHALL return an agent with Q-learning/SARSA tabular updates or a Deep Q-Network with replay memory and target network

### Requirement: Policy-gradient and actor-critic agents
The system SHALL provide policy-gradient and on-policy actor-critic agents. (src: runtime/toolbox/rl/rl_classdefs.m) (src: runtime/toolbox/rl/runtime_rl.cpp)

#### Scenario: Construct a policy-gradient agent
- **WHEN** a program constructs `rlPGAgent`, `rlPPOAgent`, or `rlTRPOAgent`
- **THEN** the system SHALL return a REINFORCE (softmax policy), PPO (clipped surrogate + GAE), or TRPO (natural gradient + KL trust region) agent

### Requirement: Continuous-control and critic-free agents
The system SHALL provide continuous-control agents and a critic-free GRPO agent. (src: runtime/toolbox/rl/rl_classdefs.m) (src: runtime/toolbox/rl/runtime_rl.cpp)

#### Scenario: Construct a continuous-control agent
- **WHEN** a program constructs `rlDDPGAgent`, `rlTD3Agent`, `rlSACAgent`, or `rlGRPOAgent`
- **THEN** the system SHALL return a DDPG (deterministic actor + Q critic), TD3 (twin critics + delayed updates), SAC (max-entropy squashed-Gaussian), or group-relative critic-free GRPO agent

### Requirement: Environments and specs
The system SHALL provide predefined and custom environments with observation/action specs. (src: runtime/toolbox/rl/runtime_rl.cpp)

#### Scenario: Create an environment
- **WHEN** a program calls `rlPredefinedEnv` (BasicGridWorld, CartPole-Discrete, Pendulum-Continuous, Countdown-Discrete), `rlMDPEnv`, or `rlFunctionEnv`, with `rlNumericSpec`/`rlFiniteSetSpec`
- **THEN** the system SHALL return an environment object usable for training and simulation

### Requirement: Training and simulation API
The system SHALL provide a unified training and simulation API with options. (src: runtime/toolbox/rl/runtime_rl.cpp) (src: runtime/toolbox/rl/rl_classdefs.m)

#### Scenario: Train and deploy a policy
- **WHEN** a program calls `train(agent, env, rlTrainingOptions(...))`, then `sim` or `getAction`/`getMaxQValue`/`getGreedyPolicy`/`getCritic`/`getLearnableParameters`
- **THEN** the system SHALL dispatch to the agent-specific training loop and return the trained agent, rollouts, and policy/critic accessors
