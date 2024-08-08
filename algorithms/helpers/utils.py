from torch import nn
from torch.nn import functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


class FinalStateLoss(nn.Module):
    def __init__(self):
        super(FinalStateLoss, self).__init__()

    def forward(self, T, final_states):

        # apply softmax to T to get better approximation
        T = nn.functional.softmax(T, dim=-1)
        loss = 0

        for state1 in final_states:
            for transition in range(T.shape[0]):
                stateProbs = 0
                for state2 in range(T.shape[1]):
                    if state1 == state2:
                        continue

                    stateProbs += T[transition][state1][state2]

                loss += stateProbs
        return loss


class NoOtherStartingStatesLoss(nn.Module):
    def __init__(self):
        super(NoOtherStartingStatesLoss, self).__init__()

    def forward(self, T, starting_states):

        # apply softmax to T to get better approximation
        T = nn.functional.softmax(T, dim=-1)
        loss = 0

        for state2 in starting_states:
            for transition in range(T.shape[0]):
                stateProbs = 0
                for state1 in range(T.shape[1]):
                    if state1 == state2:
                        continue

                    stateProbs += T[transition][state1][state2]

                loss += stateProbs
        return loss


# ToDo - this computation does only work for binary classifcation for now
def compute_class_weights(loader):

    weights = {}

    for data in loader:
        key = data.y.shape[0]

        # check if key is in weights
        if key not in weights.keys():
            weights[key] = []

        for value in data.y:
            weights[key].append(value)

    class_weights = {}

    for key, value in weights.items():
        zeros = len([v for v in value if v == 0])
        ones = len([v for v in value if v == 1])

        weight_0 = 1 - zeros / len(value)
        weight_1 = 1 - ones / len(value)
        class_weights[key] = [weight_0, weight_1]

    return class_weights


def get_pred_int(pred_score):
    if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
        return (pred_score > 0.5).long()
    else:
        return pred_score.max(dim=1)[1]
