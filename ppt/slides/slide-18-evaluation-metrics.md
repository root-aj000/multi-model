# Slide 18 — Evaluation Metrics

## What to Say (Speaker Notes)
"How do we know if the model is good? We need **metrics**. I track **17+ distinct metric types** across two levels. At the **overall level**, I compute macro accuracy (average accuracy across all attributes), weighted accuracy (weighted by class frequency), and total sample count. At the **per-attribute level** — for each of the 9 attributes — I compute accuracy, precision (both macro and weighted), recall (both macro and weighted), F1 score (both macro and weighted), the confusion matrix, and per-class breakdowns. All these metrics are logged to a single `results/results.json` file after training. This comprehensive evaluation gives us a **complete picture** of model performance — not just one number. For example, macro accuracy treats all classes equally, while weighted accuracy accounts for class imbalance. Both are useful in different contexts."

## What to Show on Screen

```
📊 EVALUATION METRICS (17+ distinct types)

   🌍 OVERALL METRICS:
      • accuracy_macro      (avg across attributes)
      • accuracy_weighted   (weighted by class freq)
      • total_samples

   🎯 PER-ATTRIBUTE METRICS (×9 attributes):
      • accuracy
      • precision_macro
      • precision_weighted
      • recall_macro
      • recall_weighted
      • f1_macro
      • f1_weighted
      • confusion_matrix
      • per_class_breakdown

   📁 All metrics logged to: results/results.json

   ❓ MACRO vs WEIGHTED?
      • MACRO   → treats all classes equally
      • WEIGHTED → accounts for class frequency
      • Both are useful — report both!
```

## Visual Suggestion
- Use a **two-column layout**: overall metrics on the left, per-attribute on the right.
- Add a **small icon** for each metric type.
- Mention that the JSON file is **machine-readable** — easy to parse and plot.

## Key Talking Points
- **One number is never enough** — we need multiple metrics to understand model behavior.
- **Macro vs weighted** is a common interview question — be ready to explain.
- The metrics are **logged automatically** — no manual calculation needed.
