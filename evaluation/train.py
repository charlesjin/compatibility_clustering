import torch
import numpy as np

def train(net, criterion, optimizer, epochs, trainloader, device, 
        past_epochs=0, scheduler=None):
    net.train()
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += int(labels.size(0))
            correct += int((predicted == labels).sum())

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        print('Accuracy for epoch %d: %d %%' % (past_epochs + epoch + 1, 100 * correct / total))
        if scheduler is not None:
            scheduler.step()

def test(net, testloader, device, source, target=None):
    net.eval()
    correct = 0
    total = 0
    target_misclassified = []

    with torch.no_grad():
        last_idx = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if target is not None:
                # poisoned test set
                end_idx = last_idx + len(labels)
                true_labels = \
                        torch.Tensor(testloader.dataset.true_targets[last_idx:end_idx])
                true_labels = true_labels.to(device)
                source_labels = (true_labels == source)
                last_idx = end_idx

                # which elements of the source class are incorrect classified as target class
                wrong_class = predicted == target
                target_misclassified.extend(wrong_class[source_labels])

                labels = true_labels
            else:
                # clean test set
                # which elements of the source class are correct classified as source class
                correct_class = predicted == source
                source_labels = (labels == source)
                target_misclassified.extend(correct_class[source_labels])

            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy, target_misclassified

