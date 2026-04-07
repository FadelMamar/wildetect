# Design System Strategy: The Digital Architect

## 1. Overview & Creative North Star
The "Digital Architect" is the creative North Star for this design system. It moves beyond the standard "boxes-and-borders" documentation template to create a space that feels engineered, high-fidelity, and editorial. 

Instead of relying on rigid, repetitive grids, this system uses **Sophisticated Tonal Layering** and **Intentional Asymmetry**. We treat documentation as a premium publication for developers—where white space is a functional tool for focus, and the interface recedes to let the code and logic shine. By layering surfaces rather than boxing them in, we create a sense of architectural depth that feels expansive yet structured.

## 2. Colors & Surface Logic

### The "No-Line" Rule
To achieve a high-end, bespoke feel, **1px solid borders are strictly prohibited for sectioning.** Boundaries must be defined through background color shifts or tonal transitions.
- Use `surface` (#f8f9fa) for the main canvas.
- Use `surface_container_low` (#f3f4f5) to define secondary content zones.
- Use `surface_container_lowest` (#ffffff) to elevate interactive cards or code blocks against a darker background.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers. A nested layout should follow this logic:
1. **Foundation:** `surface` (The base)
2. **Zone:** `surface_container_low` (A subtle area for a group of items)
3. **Focus Card:** `surface_container_lowest` (A white card "floating" on the low-grey zone)

### The Glass & Gradient Rule
For elements that require high visual impact—such as floating navigation or primary "Hero" calls to action:
- **Glassmorphism:** Use semi-transparent variants of `surface_container_lowest` with a `backdrop-blur` of 12px-20px. 
- **Signature Gradients:** For primary CTAs, transition from `primary` (#4500e8) to `primary_container` (#5e39ff) at a 135-degree angle. This adds "soul" and dimension that flat hex codes lack.

## 3. Typography
Our typography is a dialogue between two distinct voices: the **Authoritative Header** and the **Precise Body**.

*   **Display & Headlines (Plus Jakarta Sans):** These are the "Architectural" elements. Use `display-lg` to `headline-sm` for high-impact titles. The geometric nature of Plus Jakarta Sans provides a modern, developer-centric authority.
*   **Body & Titles (Inter):** The "Functional" element. Inter is used for `title-lg` down to `label-sm`. It is optimized for screen readability and provides the utilitarian precision required for technical documentation.

**Hierarchy Note:** Always maintain a significant contrast between `headline` and `body`. If a section title is `headline-sm`, the supporting text should jump down to `body-md` to ensure the layout feels intentional and editorial.

## 4. Elevation & Depth

### The Layering Principle
Depth is achieved through "Tonal Stacking." To elevate a card, do not reach for a shadow first; instead, place a `surface_container_lowest` (#ffffff) element on top of a `surface_container_low` (#f3f4f5) background. The subtle 1.5% difference in luminosity creates a sophisticated, natural lift.

### Ambient Shadows
When an element must "float" (e.g., a dropdown or a modal), use an **Ambient Shadow**:
- **Color:** A 6% opacity version of `on_surface` (#191c1d).
- **Blur:** 32px to 64px.
- **Spread:** -4px (to keep the shadow tight and professional).
- **X/Y Offset:** Y: 8px.

### The "Ghost Border" Fallback
If a border is required for accessibility (e.g., in a high-contrast mode or specific input fields), use a **Ghost Border**: `outline_variant` (#c9c4da) at **15% opacity**. This provides a hint of structure without interrupting the visual flow.

## 5. Components

### Buttons
*   **Primary:** Gradient fill (`primary` to `primary_container`), white text, `roundness.md` (0.75rem).
*   **Secondary:** `surface_container_high` background with `primary` text. No border.
*   **Tertiary/Ghost:** No background. `primary` text. Underline only on hover.

### Cards & Lists
*   **The Card Rule:** Forbid divider lines. Separate items using `spacing.6` (1.5rem) or by placing cards on a `surface_container_low` track.
*   **Interactive Cards:** On hover, shift the background from `surface_container_lowest` to a very subtle gradient or apply an **Ambient Shadow**.

### Input Fields
*   **Text Inputs:** Use `surface_container_low` for the fill. Use a \"Ghost Border\" that transitions to a 2px `primary` bottom-border only on focus. This mimics an \"architectural\" underline.

### SDK & API Components
*   **Language Selectors (Chips):** Use `secondary_container` with `on_secondary_container` text. Use `roundness.full` for a pill shape that contrasts against square code blocks.
*   **Code Blocks:** Use `inverse_surface` (#2e3132) for the background. Syntax highlighting should use the `primary_fixed` and `tertiary_fixed` palettes to maintain brand harmony.

## 6. Do's and Don'ts

### Do
*   **DO** use whitespace as a primary separator. If a section feels cluttered, increase the spacing from `spacing.12` to `spacing.20` rather than adding a line.
*   **DO** use `Plus Jakarta Sans` for anything that acts as a \"Label\" or \"Heading\" to reinforce the brand's premium feel.
*   **DO** use asymmetrical layouts for landing pages—for example, a left-aligned hero title with an overlapping \"glass\" code snippet on the right.

### Don't
*   **DON'T** use 100% black (#000000) for text. Always use `on_surface` (#191c1d) for better long-form reading comfort.
*   **DON'T** use standard Material Design \"Drop Shadows.\" They are too heavy for this \"Digital Architect\" aesthetic. Stick to **Ambient Shadows**.
*   **DON'T** put a border around code snippets or cards. Let the surface color change define the edge.
