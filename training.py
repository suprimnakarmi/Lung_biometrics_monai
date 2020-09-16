from models import model_unet

def train():
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
            outputs=model_unet(inputs)
            loss = loss_function(ouputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds)// train_loader.batch_size