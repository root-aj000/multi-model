# Slide 3 — Problem Statement

## What to Say (Speaker Notes)
"Let me start with the **problem**. Imagine you work at an advertising agency. Every single day, your team receives **thousands of new ad images** — for products, services, festivals, apps, and so on. Before these ads can be shown to people online, someone has to **manually tag** each ad with information like: *What is the theme? What emotion does it create? Is it safe? Who is the target audience?* Doing this by hand is **slow, expensive, and inconsistent** — different people tag the same ad differently. So the question is: **can we teach a computer to do this tagging automatically?** That is exactly what my project solves. The computer looks at both the **image** and the **text inside the image** (extracted using OCR), and outputs **9 structured labels** in a single shot."

## What to Show on Screen

```
❓ THE PROBLEM

• Ad agencies receive THOUSANDS of creatives every day.

• Each ad needs to be tagged with:
    – Theme, Sentiment, Emotion, Colour, Audience, CTR, ...
    – That's 9 different attributes per ad!

• Manual tagging is:
    ✗ Slow (hours per batch)
    ✗ Expensive (needs human reviewers)
    ✗ Inconsistent (different people, different tags)

• We need an AUTOMATED system that:
    ✓ Looks at the IMAGE
    ✓ Reads the TEXT inside the image (OCR)
    ✓ Outputs 9 STRUCTURED labels per ad
    ✓ Plugs into analytics dashboards
```

## Visual Suggestion
- Show a **split image**: on the left, a tired human reviewer with a stack of ads; on the right, a happy computer screen with structured labels.
- Use a **red** color for the "manual" side and a **green** color for the "automated" side.

## Key Talking Points
- Emphasize the **scale** of the problem (thousands of ads per day).
- Emphasize that **both image AND text** matter — text inside an ad is part of the creative.
- This sets up the **multi-modal** idea that comes in the next slides.
