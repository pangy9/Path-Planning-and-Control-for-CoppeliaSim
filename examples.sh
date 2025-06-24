#!/bin/bash
# --- 测试组 1: 差速驱动小车 (DifferentialDrive) ---
echo "--- 测试组 1: 差速驱动小车 (DifferentialDrive) ---"

echo "测试 1.1: AStar + PurePursuit + PI (经典组合)"
python main.py --vehicle_model DifferentialDrive --planner AStar --angular PurePursuit --speed PI

echo "测试 1.2: AStar + Stanley + PI (检验Stanley控制器)"
python main.py --vehicle_model DifferentialDrive --planner AStar --angular Stanley --speed PI

echo "测试 1.3: RRT + PurePursuit + PI (检验RRT规划器)"
python main.py --vehicle_model DifferentialDrive --planner RRT --angular PurePursuit --speed PI

echo "测试 1.4: AStar + MPC (MPC控制角度，PI控制速度)"
python main.py --vehicle_model DifferentialDrive --planner AStar --angular MPC --speed PI

echo "测试 1.5: AStar + MPC (MPC同时控制角度和速度)"
python main.py --vehicle_model DifferentialDrive --planner AStar --angular MPC --speed MPC


# --- 测试组 2: 阿克曼转向小车 (AckermannSteering) ---
echo "--- 测试组 2: 阿克曼转向小车 (AckermannSteering) ---"

echo "测试 2.1: AStar + PurePursuit + PI (阿克曼经典组合)"
python main.py --vehicle_model AckermannSteering --planner AStar --angular PurePursuit --speed PI

echo "测试 2.2: AStar + Stanley + PI (检验Stanley在阿克曼上的效果)"
python main.py --vehicle_model AckermannSteering --planner AStar --angular Stanley --speed PI

echo "测试 2.3: RRTStar + Stanley + PI (检验RRT*与阿克曼的配合)"
python main.py --vehicle_model AckermannSteering --planner RRTStar --angular Stanley --speed PI

echo "测试 2.4: AStar + MPC (MPC控制角度，PI控制速度)"
python main.py --vehicle_model AckermannSteering --planner AStar --angular MPC --speed PI

echo "测试 2.5: AStar + MPC (MPC同时控制角度和速度)"
python main.py --vehicle_model AckermannSteering --planner AStar --angular MPC --speed MPC


