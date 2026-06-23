# Slide 23 — Web Application (Frontend)

## What to Say (Speaker Notes)
"To make the model accessible to **non-technical users**, I built a **web dashboard** using **Next.js 16** with **React 19** and **TypeScript**. The styling uses **Tailwind CSS v4**. For state management, I use **Zustand** — a lightweight alternative to Redux. For charts and visualizations, I use **Recharts**. The backend client uses **Supabase JS** and **Axios** for HTTP requests. The main feature is simple: the user **uploads an ad image**, and the dashboard displays the **9 predicted attributes** as cards and bar charts. There's also a **dark/light theme toggle** using `next-themes`. The UI is responsive and works on both desktop and mobile. This makes the model usable by marketers and analysts who don't want to write code."

## What to Show on Screen

```
🖥️ WEB APPLICATION (Frontend)

   Tech Stack:
   ┌──────────────────┬────────────────────────────┐
   │ Framework        │ Next.js 16                 │
   │ UI Library       │ React 19                   │
   │ Language         │ TypeScript                 │
   │ Styling          │ Tailwind CSS v4            │
   │ State Management │ Zustand                    │
   │ Charts           │ Recharts                   │
   │ HTTP Client      │ Axios + Supabase JS        │
   │ Theme            │ next-themes (dark/light)   │
   └──────────────────┴────────────────────────────┘

   🎨 MAIN FEATURES:
      • Upload an ad image (drag & drop)
      • See 9 predicted attributes as cards
      • View confidence scores as bar charts
      • Dark / light theme toggle
      • Responsive (desktop + mobile)

   👥 USERS:
      Marketers, analysts, non-technical staff
      who don't want to write code
```

## Visual Suggestion
- **Embed a screenshot** of the actual web dashboard (upload page + results page).
- Use **callout boxes** for each tech stack component.
- Add a **small phone mockup** to show mobile responsiveness.

## Key Talking Points
- The frontend is **separate from the backend** — they communicate via the REST API.
- **Zustand** is much simpler than Redux for small apps.
- **Recharts** makes it easy to render beautiful charts with minimal code.
