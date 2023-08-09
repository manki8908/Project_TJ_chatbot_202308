def calc_acc1(out, label):

    import numpy as np

    run_acc = []

    for i in range(np.size(out,1)):

        top = ( abs(out[:,i,:] - label[:,i,:])/label[:,i,:] ).sum()
        bottom = np.size(out,0)
        run_acc.append(top/bottom)

    run_acc = np.array(run_acc)

    acc = run_acc.sum()/(np.size(out,1))

    return acc


def calc_acc2(out, label):

    import numpy as np


    _, predicted = torch.max(out, 1)
    total += label.size(1)
    correct += (predicted == label).sum()

    acc = correct/total

    return acc

