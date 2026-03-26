
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Control.IK.IK_Hubert import ik_hubert
from Control.FK.FK_Hubert import fk_hubert

def test_kinematics_cycle():
    """
    only works for theta 1 in [0,180], theta2 in [0,180] and theta3 in [0, 90]. Dont know why... doesnt matter(?)
    """
    n_of_samples = 10000
    correct_count = 0
    pos_tolerance = 1e-5
    solved_once_count  =0 
    combination_of_angles_solves_total = 0
    # for _ in range(n_of_samples):
        
    #     theta1_orig = np.random.uniform(0, 180)
    #     theta2_orig = np.random.uniform(0, 180)
    #     theta3_orig = np.random.uniform(0, 90)

    theta1_range = np.arange(0, 181, 10)
    theta2_range = np.arange(0, 181, 10)
    theta3_range = np.arange(0, 91, 10)

    workspace_points = []

    n_of_samples = len(theta1_range) * len(theta2_range) * len(theta3_range)

    for theta1_orig in theta1_range:
        print(f"Theta 1: {theta1_orig}/180")
        for theta2_orig in theta2_range:
            for theta3_orig in theta3_range:
        
                x_orig, y_orig, z_orig = fk_hubert(theta1_orig, theta2_orig, theta3_orig)
                solutions_t2_t3, solutions_t1 = ik_hubert(x_orig, y_orig, z_orig)

                solved_once = False        
                combination_of_angles_solves = 0
                for t1_pred in solutions_t1:
                    for t2_pred, t3_pred in solutions_t2_t3:
                        x_after_ik, y_after_ik, z_after_ik = fk_hubert(t1_pred, t2_pred, t3_pred)

                        if (abs(x_after_ik - x_orig) <= pos_tolerance and
                            abs(y_after_ik - y_orig) <= pos_tolerance and
                            abs(z_after_ik - z_orig) <= pos_tolerance):
                            combination_of_angles_solves +=1
                            combination_of_angles_solves_total +=1
                            solved_once = True
                
        
        
                if len(solutions_t1) > 0 and len(solutions_t2_t3) > 0 and combination_of_angles_solves == len(solutions_t1) * len(solutions_t2_t3):
                    correct_count += 1

                if solved_once: 
                    solved_once_count += 1

           
    print(f"samples = {n_of_samples}\n")
    print(f"All combinations of angles correct {correct_count}/{n_of_samples} = {correct_count/n_of_samples*100} % .")
    print(f"\natleast one combinations of angles correct in:{solved_once_count}/{n_of_samples} = {solved_once_count/n_of_samples*100} samples % \n")
    print(f"{combination_of_angles_solves_total/n_of_samples} combinations of angles per sample correct")
if __name__ == "__main__":
    test_kinematics_cycle()
