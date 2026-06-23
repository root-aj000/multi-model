# Slide 17 — Loss Function

## What to Say (Speaker Notes)
"The loss function is what the model tries to minimize during training. Since we have **9 classification heads**, we use **9 separate CrossEntropyLoss** functions — one per attribute. Each loss has **label smoothing 0.2** applied, which means instead of teaching the model to be 100% confident on the correct class, we teach it to be 80% confident on the correct class and distribute the remaining 20% across the other classes. This prevents overfitting and improves calibration. The 9 losses are then **weighted and summed** using the per-attribute loss weights I mentioned earlier. To handle **class imbalance** — for example, if there are 1000 'red' ads but only 50 'purple' ads — I compute **inverse-frequency class weights** automatically. These weights make the model pay more attention to rare classes. The total loss is what gets back-propagated through the entire network."

## What to Show on Screen

```
📉 LOSS FUNCTION

   For each of the 9 attributes:
       loss_i = CrossEntropyLoss(
                   label_smoothing = 0.2,
                   class_weights   = inverse_frequency
                )(logits_i, labels_i)

   Total loss:
       L = Σ  w_i × loss_i     for i = 1..9
           ─────────────────
           w_i = ATTRIBUTE_LOSS_WEIGHTS[i]

   ❓ WHY MULTI-TASK LOSS?
      • Shared representation across attributes
      • Acts as a REGULARIZER (can't overfit to one task)
      • Single forward pass, single backward pass

   ⚖️  CLASS IMBALANCE HANDLING:
      • Inverse-frequency weights
      • Rare classes get higher weight
      • Auto-computed from training set
```

## Visual Suggestion
- Show the **formula** prominently in the center.
- Add a **small bar chart** showing class imbalance (e.g., 1000 red vs 50 purple).
- Use a **mathematical notation** style for the loss equation.

## Key Talking Points
- **Label smoothing** is a simple but powerful trick — it makes the model less overconfident.
- **Class weights** ensure the model doesn't ignore rare classes.
- The **weighted sum** lets us prioritize some attributes over others.
