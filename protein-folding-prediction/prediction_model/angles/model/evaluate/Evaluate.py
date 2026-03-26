import torch

def EvaluateNNAngles(inputTensorsTest, targetTensorsTest,  model, device): 
    model.to(device)
    predictions_list = []
    predictions_np_list = []
    rmse_list =[]
    model.eval()
    with torch.no_grad(): 
        for i in range(inputTensorsTest.shape[0]): 
            
            inputTensor = inputTensorsTest[i, :].to(device).unsqueeze(0)#Add batch dim
            prediction = model(inputTensor) #[1, L, 1]
            prediction = prediction.squeeze() #[L, ]

            correct = targetTensorsTest[i, :].to(device)

            mask = (correct != 0).float() # True -> 1.0 and False -> 0.0
            
            diff = (prediction - correct) * mask

            nrElements = mask.sum()
            mse  = diff.pow(2).sum() / nrElements
            rmse = torch.sqrt(mse)

            rmse_list.append(rmse)

            print(f"\n \n For test protein {i+1} / {inputTensorsTest.shape[0]}, the root-mean-squared error for the angles \n was: {rmse}°")

            predictions_np = prediction.cpu().detach().numpy()
            predictions_np_list.append(predictions_np)

            predictions_list.append(prediction)
        avg_rmse = torch.stack(rmse_list).mean()
        print(f"\n average RMSE on testset mask first and last AA: {avg_rmse:.4f}°")
    return predictions_list , predictions_np_list, rmse_list
