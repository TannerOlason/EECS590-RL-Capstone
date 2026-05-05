# V3 Experiment Summary

Showing top 18 of 18 experiments, ranked by max leftmost progress.

| experiment | mean_reward | std_reward | mean_scroll | mean_max_scroll | mean_max_leftmost_world_x | mean_enemies_killed | mean_squad_alive_at_end | mean_right_move_rate | mean_idle_rate | mean_invalid_move_rate | mean_shielded_invalid_move_rate | mean_attack_intent_rate | mean_scroll_locked_by_enemy_steps | mean_scroll_locked_by_lagging_player_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260504-153923_sac_resetfix_invalid_features_250k | 20.877 | 28.217 | 32.500 | 32.500 | 39.700 | 1.000 | 1.600 | 0.641 | 0.157 | 0.458 |  | 0.951 | 150.300 | 48.100 |
| baseline_random | -5.181 | 2.270 | 18.400 |  |  | 0.200 | 0.800 |  |  |  |  |  |  |  |
| sac_500k_lockclear_combat_lean_p0p5_k2p0 | 35.326 | 30.227 | 17.400 |  |  | 0.400 | 2.500 |  |  |  |  |  |  |  |
| sac_500k_lockclear_combat_high_p0p25_k3p0 | 41.221 | 22.240 | 15.200 |  |  | 0.500 | 2.100 |  |  |  |  |  |  |  |
| sac_500k_curriculum | 55.703 | 19.133 | 13.300 |  |  | 0.200 | 2.400 |  |  |  |  |  |  |  |
| sac_1000k_high_leftmost | -20.351 | 18.938 | 8.000 | 8.000 | 10.700 | 0.500 | 2.400 | 0.805 | 0.048 |  |  | 0.245 | 9.200 | 0.000 |
| sac_500k_lockclear_balanced_p0p7_k1p5 | 37.869 | 10.656 | 7.900 |  |  | 0.500 | 2.400 |  |  |  |  |  |  |  |
| sac_500k_lockclear_combat_mid_p0p35_k2p5 | 32.018 | 2.062 | 3.600 |  |  | 0.000 | 2.900 |  |  |  |  |  |  |  |
| 20260504-123248_sac_invalid_move_features_1m | -55.516 | 40.681 | 0.600 | 0.600 | 3.000 | 0.100 | 2.600 | 0.113 | 0.466 | 0.369 |  | 0.089 | 2.100 | 0.000 |
| sac_50k_quick | -7.482 | 0.481 | 1.800 |  |  | 0.000 | 2.600 |  |  |  |  |  |  |  |
| sac_500k_lockclear_combat_max_p0p15_k4p0 | 37.547 | 11.509 | 1.500 |  |  | 0.100 | 3.000 |  |  |  |  |  |  |  |
| sac_1000k_lockclear_high_safe_scroll | 7.469 | 38.319 | 1.300 |  |  | 0.200 | 3.000 |  |  |  |  |  |  |  |
| sac_3k_smoke | -7.612 | 0.338 | 0.000 |  |  | 0.000 | 2.700 |  |  |  |  |  |  |  |
| sweep_smoke_balanced_p0p7_k1p5 | -0.092 | 0.000 | 0.000 |  |  | 0.000 | 3.000 |  |  |  |  |  |  |  |
| sweep_smoke_combat_high_p0p25_k3p0 | -0.092 | 0.000 | 0.000 |  |  | 0.000 | 3.000 |  |  |  |  |  |  |  |
| sweep_smoke_combat_lean_p0p5_k2p0 | -0.092 | 0.000 | 0.000 |  |  | 0.000 | 3.000 |  |  |  |  |  |  |  |
| sweep_smoke_combat_max_p0p15_k4p0 | -0.092 | 0.000 | 0.000 |  |  | 0.000 | 3.000 |  |  |  |  |  |  |  |
| sweep_smoke_combat_mid_p0p35_k2p5 | -0.092 | 0.000 | 0.000 |  |  | 0.000 | 3.000 |  |  |  |  |  |  |  |
