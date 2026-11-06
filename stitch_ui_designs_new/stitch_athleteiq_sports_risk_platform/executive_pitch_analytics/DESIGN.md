---
name: Executive Pitch Analytics
colors:
  surface: '#0d131f'
  surface-dim: '#0d131f'
  surface-bright: '#333947'
  surface-container-lowest: '#080e1a'
  surface-container-low: '#161c28'
  surface-container: '#1a202c'
  surface-container-high: '#242a37'
  surface-container-highest: '#2f3542'
  on-surface: '#dde2f4'
  on-surface-variant: '#bacbb7'
  inverse-surface: '#dde2f4'
  inverse-on-surface: '#2a303e'
  outline: '#859582'
  outline-variant: '#3c4b3b'
  surface-tint: '#04e55e'
  primary: '#7fff91'
  on-primary: '#003911'
  primary-container: '#10e760'
  on-primary-container: '#006223'
  inverse-primary: '#006e29'
  secondary: '#b7c7ea'
  on-secondary: '#20304d'
  secondary-container: '#3a4967'
  on-secondary-container: '#a9b9db'
  tertiary: '#dde4fe'
  on-tertiary: '#283043'
  tertiary-container: '#c1c8e1'
  on-tertiary-container: '#4c5468'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#6aff85'
  primary-fixed-dim: '#04e55e'
  on-primary-fixed: '#002107'
  on-primary-fixed-variant: '#00531d'
  secondary-fixed: '#d7e2ff'
  secondary-fixed-dim: '#b7c7ea'
  on-secondary-fixed: '#091b36'
  on-secondary-fixed-variant: '#374764'
  tertiary-fixed: '#dbe2fb'
  tertiary-fixed-dim: '#bfc6df'
  on-tertiary-fixed: '#131b2d'
  on-tertiary-fixed-variant: '#3f465b'
  background: '#0d131f'
  on-background: '#dde2f4'
  surface-variant: '#2f3542'
typography:
  headline-xl:
    fontFamily: Space Grotesk
    fontSize: 40px
    fontWeight: '600'
    lineHeight: 48px
  headline-xl-mobile:
    fontFamily: Space Grotesk
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 30px
    fontWeight: '600'
    lineHeight: 38px
  headline-lg-mobile:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 22px
    fontWeight: '500'
    lineHeight: 30px
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 26px
  kpi-stat:
    fontFamily: Space Grotesk
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
  body-lg:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 26px
  body-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 22px
  body-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 16px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-mobile: 1rem
  margin: 2.5rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system targets sports executives, general managers, performance directors, and elite tier quantitative analysts. The product posture is surgical, uncompromised, and calm: an ultra-refined command interface engineered to eliminate data fatigue while maintaining absolute analytical depth.

The design movement combines **Minimalism** with high-precision **Technical SaaS Architecture**. Rather than overwhelming decision-makers with saturated charts and aggressive dashboards, the system prioritizes negative space, crisp geometric containers, and quiet legibility. High-energy electric lime is treated like an illuminated scalpel: reserved exclusively for high-conviction signals, active prediction outcomes, and key tactical interactions.

## Colors

The palette leverages an engineered deep slate continuum to separate information levels without visual weight or heavy shadows.

### Core Architecture
- **Canvas Base (`#080E1A`)**: Foundation backdrop delivering deep visual stability and minimum glare during late-night operational reviews.
- **Surface Level 1 (`#0D1527`)**: Sub-panels, toolbars, sidebar scaffolds, and data table headers.
- **Surface Level 2 / Card Background (`#131E33`)**: Interactive tiles, analytical widgets, and elevated modal panels.
- **Hairline Border (`#1E2E4A`)**: Structural definition lines for cards, cell divides, and chart grids.
- **Surface Highlight Hover (`#192843`)**: Hover state fill for list items and table rows.

### Signal & Focal Tokens
- **Primary Energy (`#10E760`)**: Electric Lime. Used strictly for optimal performance metrics, active statistical models, primary call-to-action buttons, and confidence indicators above 80%. Never used as decorative canvas fills.
- **Risk & Alert Accents**:
  - Positive / Delta Up: `#10E760` (Emerald Lime)
  - Neutral / Variance Alert: `#F59E0B` (Refined Amber)
  - Critical Risk / Delta Down: `#FF4D4D` (Soft Coral Red)

### Typography Tones
- **Text Primary (`#F8FAFC`)**: High-contrast, optical white for primary metrics and titles.
- **Text Muted / Subtitle (`#94A3B8`)**: Mid-tier slate for body text, column headers, and labels.
- **Text Tertiary / Metadata (`#64748B`)**: Low-contrast slate for axis ticks, timestamps, and inactive iconography.

## Typography

The typographic hierarchy establishes clear operational priority across real-time telemetry:

- **Headlines & Primary Metrics (`Space Grotesk`)**: Provides an authoritative, technical, and forward-looking frame for titles and large KPI readouts. Numbers are kerned for quick cognitive processing during rapid briefings.
- **Body Content & Structural Prose (`Geist`)**: A clean, highly legible grotesque designed for dense digital screens. Delivers effortless long-form reading on complex athletic evaluation briefs.
- **Data Attributes & Status Microcopy (`JetBrains Mono`)**: Strict tabular monospaced representation for probabilities, timestamps, financial valuations, and delta badges. Ensures column alignment and tabular stability when numeric values fluctuate.

## Layout & Spacing

The interface is driven by a 12-column responsive fluid grid with generous outer margins to avoid information claustrophobia.

### Structural Parameters
- **Desktop (>= 1280px)**: 12-column layout with 24px (`1.5rem`) gutters and 40px (`2.5rem`) viewport boundary margins. Dashboard canvas maxes out at 1600px width with automatic centering.
- **Tablet (768px - 1279px)**: 8-column layout with 20px gutters and 24px canvas margins.
- **Mobile (< 768px)**: 4-column layout with 16px (`1rem`) gutters and 16px margins. High-density KPI cards convert to single-column horizontal scroll or swipeable blocks.

### Spacing Philosophy
Executive decision-making requires intentional breathing room around data clusters. Cards use a minimum internal padding of `space-lg` (24px). Spacing between unrelated analytical groups relies on `space-xl` (40px) to establish mental boundaries without relying on thick separating lines.

## Elevation & Depth

This system intentionally eliminates blurred drop-shadows and fuzzy neon glows. Instead, visual planes are differentiated through **Tonal Layering** and **Low-Contrast Structural Borders**.

- **Level 0 (Canvas Base)**: `#080E1A` with zero boundary definition.
- **Level 1 (Panels & Navbars)**: `#0D1527` bounded by a 1px border of `#1E2E4A`.
- **Level 2 (Active Cards & Tables)**: `#131E33` bounded by a 1px border of `#1E2E4A`.
- **Level 3 (Dropdowns & Modals)**: `#17233D` surrounded by a 1px border of `#2C426B` and an ambient, ultra-diffused dark tint (`rgba(2, 6, 14, 0.65)` with a 32px blur radius).
- **Chart Elevation**: Recharts line and bar containers sit directly within Level 2 cards. Grid lines are tinted to `#1E2E4A` with 50% opacity, and reference dots use thin 1px `#080E1A` negative cutouts to preserve visual separation.

## Shapes

The interface embraces a precise, modern, and architectural edge treatment. 

Corners utilize `roundedness: 1` (`0.25rem` or 4px for small elements, scaling to `0.5rem` or 8px for standard cards and major overlays). Fully circular radii are restricted exclusively to numerical status pips, status avatar circles, and subtle iconography housings. Soft or pill-shaped containers are strictly forbidden to preserve an authoritative, enterprise feel.

## Components

### Buttons
- **Primary Action**: Solid `#10E760` fill with `#080E1A` bold text. Corner radius: 4px. Focus states display an offset 2px ring in `#10E760`. Reserved exclusively for primary commits (e.g., "Run Simulation", "Deploy Model").
- **Secondary Action**: Background `#131E33`, 1px border `#1E2E4A`, text `#F8FAFC`. On hover, border shifts to `#2C426B` and background to `#192843`.
- **Ghost Action**: Transparent background, text `#94A3B8`. Hover transitions text to `#F8FAFC` and applies a subtle `#0D1527` background.

### Cards & KPI Tiles
- Base fill: `#131E33` with 1px border in `#1E2E4A`.
- Internal padding: 24px (`space-lg`).
- KPI Layout: Small uppercase mono label (`JetBrains Mono`, 11px, `#94A3B8`) followed by the metric (`Space Grotesk`, 36px, `#F8FAFC`). Accompanying delta tags sit alongside or directly below the main stat.

### Badges & Prediction Chips
- Fixed mono font (`JetBrains Mono`, 11px, medium weight).
- Border radius: 4px. Padding: 2px 8px.
- Positive / Active: Background `rgba(16, 231, 96, 0.08)`, border `rgba(16, 231, 96, 0.25)`, text `#10E760`.
- Warning / Watchlist: Background `rgba(245, 158, 11, 0.08)`, border `rgba(245, 158, 11, 0.25)`, text `#F59E0B`.
- Risk / Depleted: Background `rgba(255, 77, 77, 0.08)`, border `rgba(255, 77, 77, 0.25)`, text `#FF4D4D`.

### Tables
- Header: `#0D1527` background, text `#64748B`, 12px, tracking-wide mono. Height: 44px.
- Body Rows: Height: 52px. Horizontal border only: 1px `#1E2E4A`. Alternate row striping is omitted.
- Hover State: Entire row highlights to `#192843` with zero shift in border styles.
- Cell Text: Primary identifiers in `#F8FAFC`, numerical metrics in `#94A3B8` (monospaced, right-aligned).

### Form Controls & Inputs
- Container: Background `#0D1527`, 1px solid border `#1E2E4A`, text `#F8FAFC`, corner radius: 4px.
- Focus State: Border transitions cleanly to `#10E760` with no external box-shadow glow.
- Checkboxes: 16x16px squares with 1px border `#1E2E4A`. Checked state uses `#10E760` fill with `#080E1A` check icon.

### Recharts & Data Visualization Guidelines
- **Tooltip**: Custom HTML tooltip using background `#0D1527`, 1px border `#1E2E4A`, padding 12px, font `Geist` and `JetBrains Mono`.
- **Chart Palette**:
  - Series Primary: `#10E760` (Solid 2px stroke or clean fill)
  - Series Comparison: `#64748B` (Dashed or muted solid)
  - Accent Forecast: `#38BDF8` (Precision Cyan)
- **Grid Lines**: Horizontal lines only, `#1E2E4A` with strokeDasharray `3 3`. Zero vertical lines.