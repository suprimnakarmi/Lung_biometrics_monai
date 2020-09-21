import torch
import os
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice


from models import model_unet, device
#from main import optimizer, loss_function


def train(epoch_num, val_interval, train_loader, val_loader, writer, train_ds, loss_function, optimizer):
    best_metric = -1
    best_metric_epoch = -1
    total_train = 0
    correct_train = 0
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(epoch_num):
        print("-"*10)
        print(f"epoch {epoch+1}/{epoch_num}")
        model_unet.train()
        epoch_loss=0
        step = 0
        for batch_data in train_loader:
            step +=1
            inputs, labels =(
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model_unet(inputs)
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds)// train_loader.batch_size

            # Check the accuracy for training dataset
            _, predicted = torch.max(outputs.data,1)
            total_train += labels.nelement()
            correct_train += predicted.eq(torch.squeeze(labels.data,1)).sum().item()
            train_accuracy = correct_train/ total_train

            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, train_accuracy: {train_accuracy:.4f}")
            writer.add_scalar("train_loss",loss.item(),epoch_len* epoch+step)
            writer.add_scalar("train_accuracy", train_accuracy, epoch_len* epoch+step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch{epoch +1} average loss: {epoch_loss:.4f}")

        if(epoch+1)% val_interval==0:
            model_unet.eval()
            metric_sum=0.0
            metric_count= 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                # Calculating the validation loss
                v_outputs= model_unet(val_inputs)
                loss_val = loss_function(v_outputs, val_labels)
                print(f"validation_loss: {loss_val.item()}")
                writer.add_scalar("validation_loss", loss_val.item(), epoch_len* epoch + step)

                roi_size=(160,160)
                sw_batch_size= 4
                val_outputs = sliding_window_inference(val_inputs,roi_size, sw_batch_size, model_unet)
                value = compute_meandice(
                    y_pred = val_outputs,
                    y = val_labels,
                    include_background = False,
                    to_onehot_y = True,
                    mutually_exclusive = True,
                )
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            writer.add_scalar("val_mean_dice", metric, epoch+1)
            metric_values.append(metric)
            if metric>best_metric:
                best_metric = metric
                best_metric_epoch = epoch +1
                torch.save(model_unet.state_dict(), os.path.join('/home/suprim/dataset/MontgomerySet/best_metric',"best_metric_model_10.pth"))
                print("saved new best metric model")
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                      f"\n best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
