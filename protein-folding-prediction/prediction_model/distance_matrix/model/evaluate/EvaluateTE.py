import torch

def EvaluateDM(inputTensorsTest, msa_tensors,targetTensorsTest,  model, device): 
    model.to(device)
    predictions_list = []
    predictions_np_list = []
    rmse_list =[]
    model.eval() #no dropout...
    with torch.no_grad(): 
        for i in range(inputTensorsTest.shape[0]): 
            inputTensor = inputTensorsTest[i, :].to(device)
            msa_tensor = msa_tensors[i].to(device)
            prediction = model(inputTensor.unsqueeze(0), msa_tensor.unsqueeze(0)) #Add batch dim
            
            correct = targetTensorsTest[i, :].to(device)
            
            maskDiagonal = torch.eye(correct.shape[1], device=device)  
            prediction = prediction.squeeze(0)
            
            mask_data = (correct > 0).float()           # 1 på data, 0 på padding
            valid_mask = mask_data * (1 - maskDiagonal)    # 1 på giltiga icke‑diagonal‑element

            prediction = prediction * valid_mask
            diff2 = (prediction - correct).pow(2) 
            mse   = diff2.sum() / valid_mask.sum()
            rmse  = torch.sqrt(mse)

            rmse_list.append(rmse)

            print(f"\n \n For test protein {i+1} / {inputTensorsTest.shape[0]}, the root-mean-squared error for the distances \n in the distance matrix was: {rmse} Å")

            predictions_np = prediction.cpu().detach().numpy()
            predictions_np_list.append(predictions_np)
            predictions_list.append(prediction)

        avg_rmse = torch.stack(rmse_list).mean()
        print(f"\n average RMSE  {avg_rmse:.4f} Å")
    return predictions_list , predictions_np_list, rmse_list
