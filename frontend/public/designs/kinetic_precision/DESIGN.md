---
name: Kinetic Precision
colors:
  surface: '#051424'
  surface-dim: '#051424'
  surface-bright: '#2c3a4c'
  surface-container-lowest: '#010f1f'
  surface-container-low: '#0d1c2d'
  surface-container: '#122131'
  surface-container-high: '#1c2b3c'
  surface-container-highest: '#273647'
  on-surface: '#d4e4fa'
  on-surface-variant: '#bacbb7'
  inverse-surface: '#d4e4fa'
  inverse-on-surface: '#233143'
  outline: '#859582'
  outline-variant: '#3c4b3b'
  surface-tint: '#04e55e'
  primary: '#7fff91'
  on-primary: '#003911'
  primary-container: '#10e760'
  on-primary-container: '#006223'
  inverse-primary: '#006e29'
  secondary: '#b5c7ed'
  on-secondary: '#1e304f'
  secondary-container: '#354767'
  on-secondary-container: '#a3b5db'
  tertiary: '#c9e9ff'
  on-tertiary: '#00354a'
  tertiary-container: '#82d2ff'
  on-tertiary-container: '#005a7b'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#6aff85'
  primary-fixed-dim: '#04e55e'
  on-primary-fixed: '#002107'
  on-primary-fixed-variant: '#00531d'
  secondary-fixed: '#d7e3ff'
  secondary-fixed-dim: '#b5c7ed'
  on-secondary-fixed: '#061b39'
  on-secondary-fixed-variant: '#354767'
  tertiary-fixed: '#c4e7ff'
  tertiary-fixed-dim: '#7bd0ff'
  on-tertiary-fixed: '#001e2c'
  on-tertiary-fixed-variant: '#004c69'
  background: '#051424'
  on-background: '#d4e4fa'
  surface-variant: '#273647'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-xl:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.015em
  headline-lg:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.005em
  metric-xl:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  metric-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.04em
  data-mono:
    fontFamily: Inter
    fontSize: 13px
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
  gutter: 1rem
  gutter-lg: 1.5rem
  margin: 1.5rem
  margin-lg: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
  space-2xl: 2rem
---

## Brand & Style
The design system embodies the high-stakes discipline of elite human performance. Designed specifically for sports scientists, head athletic trainers, and performance directors, the visual language avoids consumer fitness tropes in favor of an exacting, cockpit-grade analytics environment. 

The aesthetic leverages **High-Contrast Dark Glass & Precision Biometrics**: an immersive, deep navy-charcoal environment illuminated by high-chroma electric lime accents and clinical risk indicators. The atmosphere must feel authoritative, calculated, and proactive—evoking total control over biomechanical loads, recovery thresholds, and soft-tissue injury risk. 

Surfaces utilize crisp, razor-thin structural borders and subtle backdrop filters, creating a focused visual architecture capable of sustaining long analytical sessions under varying ambient light without fatigue.

## Colors
The color foundation relies on deeply saturated midnight pigments punctuated by calibrated functional accents:

- **Canvas & Surface Architecture**:
  - `canvas-default`: `#0a0f1d` (Deepest charcoal/navy canvas)
  - `canvas-subtle`: `#0f172a` (Sidebar and secondary layout container)
  - `surface-card`: `#111c35` (Primary data container)
  - `surface-elevated`: `#151f38` (Hover states, elevated popovers, active selection)
  - `border-subtle`: `#233554` (Structural dividers and crisp 1px borders)
  - `border-strong`: `#334a75` (Active cell selection, focused input boundaries)

- **Primary Brand Accent**:
  - `primary`: `#10e760` (High-energy electric lime; primary actions, key biomechanical metrics)
  - `primary-glow`: `rgba(16, 231, 96, 0.15)` (Radiance halos on peak metric callouts)
  - `primary-foreground`: `#021407` (Deep emerald black for high-contrast text on primary buttons)

- **Status & Biomechanical Risk System**:
  - `risk-low`: `#10b981` (Vibrant emerald green; optimal tissue recovery, compliant load)
  - `risk-moderate`: `#f59e0b` (Rich amber gold; elevated acute-to-chronic workload threshold)
  - `risk-elevated`: `#f97316` (Intense athletic orange; strain anomaly detected)
  - `risk-high`: `#ef4444` (Crimson red; critical injury likelihood, immediate intervention required)

- **Text & Data Tiers**:
  - `text-primary`: `#f8fafc` (Headings, primary quantitative figures)
  - `text-secondary`: `#94a3b8` (Muted cool slate; chart axes, table headers, labels)
  - `text-muted`: `#64748b` (Units of measurement, inactive controls)

## Typography
Typography is strictly optimized for fast visual parsing and high data density. The system relies on **Inter** across all tiers, explicitly enforcing OpenType features: `font-feature-settings: 'tnum' 1, 'cv05' 1, 'zero' 1` (tabular numerals, clean lowercase l, and slashed zeros).

- **Data Readouts & Metrics**: Metrics require tabular alignment to prevent horizontal layout shift during live telemetry or biometric streaming.
- **Labels & Micro-headers**: Uppercase labels (`label-sm`) with wide tracking (`0.04em`) are dedicated to physiological tags, axes identifiers, and metric units (e.g., `ACWR`, `HRV (MS)`, `PLAYERLOAD™`).
- **Hierarchy Balance**: Numerical readouts use heavier weights (`600` and `700`) paired with light contextual units to prioritize immediate situational awareness.

## Layout & Spacing
The layout adheres to a 12-column dynamic desktop fluid grid configured for high-density monitor consoles (1440px to 2560px screen widths). 

- **Grid Architecture**: Standard column gutter is `1rem` (16px), allowing maximum chart surface area while preserving distinct boundary separation. Outer screen margin is fixed at `1.5rem` on standard displays, expanding to `2rem` on wide displays.
- **Spatial Rhythm**: Components maintain compact internal vertical padding (`space-xs` to `space-md`) to ensure critical team-wide biometric indexes fit above the fold. 
- **Analytical Splitting**: Standard dashboard screens feature a persistent 280px left rail (athlete rosters, cohort filtering), a fluid analytical core (Recharts time-series and correlation scatterplots), and an optional collapsible 340px right panel (load prescription and clinical injury notes).

## Elevation & Depth
Depth is created through tonal layering combined with low-contrast structural outlines rather than heavy drop shadows:

- **Tonal Stratification**:
  - `Layer 0 (Canvas)`: Flat base `#0a0f1d`.
  - `Layer 1 (Card Panels)`: `#111c35` with `1px solid #233554`.
  - `Layer 2 (Interactive Floating / Menus)`: `#151f38` with `1px solid #334a75` and shadow `0 8px 24px -4px rgba(2, 6, 23, 0.6)`.
- **Luminescence & Glows**: Active and critical status metrics use subtle inner or drop glow styling:
  - Low-risk/normal status uses `0 0 12px rgba(16, 231, 96, 0.25)`.
  - High-risk alert status uses `0 0 14px rgba(239, 68, 68, 0.3)`.
- **Glass Tinting**: Translucent card surfaces apply `backdrop-filter: blur(12px)` over background charts or contextual map visualizations with `rgba(17, 28, 53, 0.85)`.

## Shapes
A disciplined "Soft" curvature philosophy is applied across the system. The baseline radius is `0.25rem` (4px), scaling to `0.5rem` (8px) on standard data cards and dialogs:

- **Panels & Cards**: `0.5rem` (8px). Keeps cards sharp, precise, and modular.
- **Interactive Controls (Inputs, Buttons, Segmented Controls)**: `0.25rem` (4px). Provides a functional, technical cockpit appearance.
- **Status Badges & Metric Pills**: `9999px` (Full Pill). Exclusively reserved for status chips and risk indicators to differentiate visual metadata from clickable square buttons.

## Components

### Buttons
- **Primary Action**: Solid `#10e760` fill, text `#021407` (`Inter 600`), `0.25rem` radius, padding `0.5rem 1rem`. Hover: Brightens to `#2bf376` with a soft green edge glow. Active: Slight scale `0.98`.
- **Secondary / Ghost**: `#151f38` fill with `1px solid #233554` border, text `#f8fafc`. Hover: Border shifts to `#38bdf8`, background to `#1e2b4d`.
- **Destructive / Override**: Dark burgundy wash `rgba(239, 68, 68, 0.15)` with `1px solid #ef4444`, text `#ef4444`.

### Cards & Analytical Panels
- Dark glass surface (`#111c35`), crisp 1px border (`#233554`), `0.5rem` radius. Header areas feature an integrated top border accent or subtle border-bottom divider (`1px solid #1a2844`) separating contextual controls from data visualizations.

### Status Badges & Glowing Metric Tags
- Pill radius (`9999px`), padding `0.125rem 0.5rem`, typography `label-sm`.
- **High Risk**: Border `1px solid #ef4444`, background `rgba(239, 68, 68, 0.12)`, text `#ef4444`, paired with a pulsing `6px` circular beacon.
- **Moderate Risk**: Border `1px solid #f59e0b`, background `rgba(245, 158, 11, 0.12)`, text `#f59e0b`.
- **Low Risk**: Border `1px solid #10b981`, background `rgba(16, 185, 129, 0.12)`, text `#10b981`.

### Data Tables
- Edge-to-edge compact rows with fixed column headers (`background: #0f172a`, `border-bottom: 1px solid #233554`).
- Cells display `tabular-nums` for heart rate, acute-chronic workload ratios, and distance metrics.
- Zebra striping is omitted in favor of hover highlighting: row hover shifts background to `rgba(35, 53, 84, 0.35)` with an active indicator stripe on the left edge.

### Inputs & Tactile Filter Toggles
- Inputs use `#0b1222` recessed fill, `1px solid #233554`, `0.25rem` radius, text `#f8fafc`. Focus state commands a 1px halo in `#10e760`.
- Segmented switches (e.g., "7D | 14D | 28D | Season") utilize an inset container `#0a0f1d` with active tab surfaced at `#151f38` with `1px solid #233554` and electric lime active text.

### Recharts & Data Visualizations
- Cartesian grids use stroke `#1b2a47` (dashed `3 3`).
- Axis tick labels use `#94a3b8` (`11px Inter`).
- Tooltip popover: `#0f172a` solid container, `1px solid #334a75`, with risk-matched colored indicator squares and tabular data values.
- Workload envelope lines feature soft gradient area fills transitioning to `rgba(16, 231, 96, 0.0)`.