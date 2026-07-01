import torch


class ConfidentLearning:
    def __init__(self, n_classes_per_attribute, threshold='auto'):
        self.n_classes_per_attribute = n_classes_per_attribute
        self.threshold = threshold
        self._thresholds = None

    @staticmethod
    def compute_confident_joint(labels, probs, thresholds=None):
        N, C = probs.shape
        device = probs.device

        if thresholds is None:
            thresholds = torch.zeros(C, device=device)
            for k in range(C):
                mask = labels == k
                thresholds[k] = probs[mask, k].mean() if mask.sum() > 0 else 1.0 / C

        confident_joint = torch.zeros((C, C), device=device, dtype=torch.long)

        for i in range(N):
            y = labels[i].item()
            above = torch.where(probs[i] >= thresholds)[0]
            if above.numel() == 0:
                confident_joint[y, y] += 1
            else:
                j = above[probs[i, above].argmax()]
                confident_joint[y, j] += 1

        row_sums = confident_joint.sum(dim=1)
        for k in range(C):
            if row_sums[k] == 0:
                confident_joint[k, k] = 1

        return confident_joint

    @staticmethod
    def compute_noise_matrix(probabilities, labels):
        noise_matrices = {}
        for attr in probabilities:
            probs = probabilities[attr]
            labels_attr = labels[attr]
            cj = ConfidentLearning.compute_confident_joint(labels_attr, probs)
            col_sums = cj.float().sum(dim=0, keepdim=True).clamp(min=1)
            noise_matrix = cj.float() / col_sums
            noise_matrices[attr] = noise_matrix
        return noise_matrices

    def calibrate_thresholds(self, probs, labels):
        thresholds = {}
        for attr in probs:
            p = probs[attr]
            lbl = labels[attr]
            C = p.shape[1]
            device = p.device
            t = torch.zeros(C, device=device)
            for k in range(C):
                mask = lbl == k
                t[k] = p[mask, k].mean() if mask.sum() > 0 else 1.0 / C
            thresholds[attr] = t
        self._thresholds = thresholds
        return thresholds

    def find_label_issues(self, probabilities, labels, return_indices=True):
        if self.threshold == 'auto' and self._thresholds is None:
            self.calibrate_thresholds(probabilities, labels)

        all_issues = set()
        for attr in probabilities:
            probs = probabilities[attr]
            labels_attr = labels[attr]
            device = probs.device
            N = probs.shape[0]

            if self._thresholds is not None and attr in self._thresholds:
                t = self._thresholds[attr]
            else:
                C = probs.shape[1]
                t = torch.zeros(C, device=device)
                for k in range(C):
                    mask = labels_attr == k
                    t[k] = probs[mask, k].mean() if mask.sum() > 0 else 1.0 / C

            self_conf = probs[torch.arange(N, device=device), labels_attr]
            issues = torch.where(self_conf < t[labels_attr])[0]
            all_issues.update(issues.tolist())

        sorted_issues = sorted(all_issues)
        result = torch.tensor(sorted_issues, dtype=torch.long)
        if return_indices:
            return result
        return result


def suggest_corrections(probabilities, labels, top_k=3):
    n_classes = {attr: probs.shape[1] for attr, probs in probabilities.items()}
    cl = ConfidentLearning(n_classes)
    issues = cl.find_label_issues(probabilities, labels)

    suggestions = {}
    for idx in issues.tolist():
        idx_suggestions = {}
        for attr in probabilities:
            probs = probabilities[attr][idx]
            current_label = labels[attr][idx].item()
            sorted_probs, sorted_classes = probs.sort(descending=True)
            top_classes = []
            for p, c in zip(sorted_probs.tolist(), sorted_classes.tolist()):
                if c != current_label and len(top_classes) < top_k:
                    top_classes.append((c, p))
            idx_suggestions[attr] = top_classes
        suggestions[idx] = idx_suggestions

    return issues, suggestions
