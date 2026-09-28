---
name: Orthonym documentation
description: A Read-mode surface of the Orthonym world, where every name wears its evidence.
colors:
  canvas: "#ffffff"
  soft: "#f4f4f4"
  hair: "#d9d9d9"
  hair-strong: "#c5c5c5"
  ink: "#000000"
  body: "#333333"
  body-strong: "#1a1a1a"
  muted: "#666666"
  crimson: "#c41e3a"
  crimson-light: "#d63553"
  crimson-deep: "#9d1830"
  mark-pin: "#2f6b28"
  mark-fallback: "#556b2f"
  mark-best: "#9c4109"
  mark-abstain: "#666666"
  glow-pin: "#3dff8f"
  glow-fallback: "#b6f24a"
  glow-best: "#ffb02e"
  bench-pink: "#f6e2e6"
  bench-grey: "#e6e6e9"
  bench-cool: "#d5d8dc"
  dark-canvas: "#1b1c20"
  dark-ground: "#141518"
  dark-ink: "#f5f6f8"
  dark-body: "#dcdee2"
  dark-muted: "#a9adb4"
typography:
  display:
    fontFamily: "Saira Condensed, Arial Narrow, sans-serif"
    fontSize: "clamp(2.2rem, 3.9vw, 3.3rem)"
    fontWeight: 400
    lineHeight: 1.02
    letterSpacing: "0.16rem"
  headline:
    fontFamily: "Saira Condensed, Arial Narrow, sans-serif"
    fontSize: "clamp(2rem, 3.6vw, 3rem)"
    fontWeight: 400
    lineHeight: 1.1
    letterSpacing: "0.1875rem"
  title:
    fontFamily: "Saira Condensed, Arial Narrow, sans-serif"
    fontSize: "clamp(1.4rem, 2.2vw, 1.75rem)"
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: "0.125rem"
  subtitle:
    fontFamily: "Saira Condensed, Arial Narrow, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 400
    lineHeight: 1.3
    letterSpacing: "0.09375rem"
  body:
    fontFamily: "Public Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.7
  body-strong:
    fontFamily: "Public Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.7
  code:
    fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.86em"
    fontWeight: 400
    fontFeature: "tnum"
  label:
    fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.6875rem"
    fontWeight: 400
    letterSpacing: "0.125rem"
  nav:
    fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    letterSpacing: "0.125rem"
  wordmark:
    fontFamily: "Saira Condensed, Arial Narrow, sans-serif"
    fontSize: "1.35rem"
    fontWeight: 400
    lineHeight: 1
    letterSpacing: "0.375rem"
rounded:
  inline: "6px"
  nav-item: "8px"
  chrome: "10px"
  sm: "14px"
  card: "22px"
  bench: "36px"
  pill: "9999px"
spacing:
  seam: "0.45rem"
  sm: "0.9rem"
  md: "1.1rem"
  lg: "1.5rem"
  xl: "2.5rem"
components:
  button-try:
    backgroundColor: "{colors.crimson-deep}"
    textColor: "{colors.canvas}"
    typography: "{typography.nav}"
    rounded: "{rounded.pill}"
    padding: "0.5rem 1rem 0.5rem 1.1rem"
  button-try-hover:
    backgroundColor: "{colors.crimson}"
  card-door:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.body}"
    rounded: "{rounded.card}"
    padding: "1.15rem 1.2rem 2.4rem"
  card-ladder:
    backgroundColor: "{colors.canvas}"
    rounded: "{rounded.card}"
    padding: "0.4rem 1.6rem 1rem"
  readback-well:
    backgroundColor: "{colors.soft}"
    rounded: "{rounded.sm}"
    padding: "0.7rem 0.85rem"
  code-block:
    backgroundColor: "{colors.soft}"
    textColor: "{colors.body-strong}"
    typography: "{typography.code}"
    rounded: "{rounded.sm}"
    padding: "0.95rem 1.1rem"
  code-inline:
    backgroundColor: "{colors.soft}"
    textColor: "{colors.body-strong}"
    typography: "{typography.code}"
    rounded: "{rounded.inline}"
    padding: "0.06em 0.32em"
  api-signature:
    backgroundColor: "{colors.soft}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "0.55rem 0.8rem"
  admonition:
    backgroundColor: "{colors.canvas}"
    rounded: "{rounded.sm}"
    padding: "0.9rem 1.1rem"
  search-input:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    rounded: "{rounded.chrome}"
  nav-current:
    backgroundColor: "{colors.soft}"
    textColor: "{colors.ink}"
    rounded: "{rounded.nav-item}"
---

# Design System: Orthonym documentation

## Overview

**Creative North Star: "The Name Wears Its Evidence"**

This is the reading room of the Orthonym world. The world is set by the README pictures and the web app: a bench of pink wash fading into cool grey under an 18 px dot grid, white cards with hairline seams, three faces with separate jobs, and four tier marks where the shape carries the tier. The docs keep all of that. What changes is the mode: this is a place for reading. The bench shows up once per page, behind the landing hero. Every other page is a white canvas with long, calm prose and hairlines. Every name shown on the site carries its mark and the rule under it, and every decline carries its reason.

The palette is ink on white with a single crimson. Crimson has two places only: the TRY IT ONLINE pill and the reading light behind the wordmark. Every other colour on the site belongs to a tier. Depth is soft and ambient and only ever lifts a card off the bench. Headings are Saira Condensed in uppercase, prose is Public Sans, and anything a person would type or compare (commands, identifiers, InChIKeys) is JetBrains Mono. Dark mode follows the system and is built from the web app's on-dark tokens. In dark mode the lamp neons that only glow on white become the tier marks themselves, because the dark card gives them the contrast.

The landing page does not use the category's hero headline over three feature cards. It uses a ladder with four real names, one per tier. The top rung is opened to show both InChIKeys and the verdict.

**Key Characteristics:**
- Ink on white, one crimson used twice, colour otherwise reserved for tier marks.
- Three faces with fixed jobs: Saira caps for display, Public Sans for prose, JetBrains Mono for what is typed or compared.
- White cards on a dotted pink-to-grey bench, lifted by an ambient two-layer shadow.
- The tier grammar: ring marks (shape carries the tier) and the rule a name wears (double, dashed, dotted, thin solid).
- Hairline seams (1px) separate everything. No side bars and no heavy borders.
- Reading measure held at 72ch. Tables scroll inside themselves on narrow screens and never scroll the page.

## Colors

A monochrome ink system with one crimson and a four-colour tier vocabulary.

### Primary
- **Orthonym Crimson** (crimson, with crimson-light at the top of the gradient and crimson-deep at the bottom): only on the TRY IT ONLINE pill, as a vertical gradient from crimson-light to crimson-deep that settles to crimson over crimson-deep on hover, and as the radial reading light behind the wordmark (crimson at .26 alpha in light mode, a warmer rgba(255,69,96,.34) core in dark mode). crimson-deep is also the pill's focus outline.

### Tertiary (the tier vocabulary)
- **PIN Green** (mark-pin): the PIN mark and the verified verdict line ("Same molecule"). On dark it becomes glow-pin.
- **Fallback Olive** (mark-fallback): the fallback mark. On dark it becomes glow-fallback.
- **Best-effort Amber** (mark-best): the best-effort mark. On dark it becomes glow-best.
- **Abstain Grey** (mark-abstain, the same value as muted): the no-name mark.
- **Lamp Neons** (glow-pin, glow-fallback, glow-best): on white they only glow and are never drawn as a line or as text. The lit PIN mark on the landing ladder has a drop-shadow halo of glow-pin at 80%. In dark mode they are the mark colours.

### Neutral
- **Canvas White** (canvas): the page, every card, the admonition surface.
- **Soft Grey** (soft): wells inside cards (the read-back), code blocks, inline code, API signatures, the current sidebar item, the round-trip verdict row, the internal-module note.
- **Hairline** (hair): every 1px seam, including rung dividers, table rows, card borders, the header underline and the API body rule.
- **Strong Hairline** (hair-strong): the table header rule, resting link underlines, the scrollbar thumb, the abstain rule.
- **Ink** (ink): headings, links, focus rings, selection background, the pin and fallback rules.
- **Body** (body) and **Body Strong** (body-strong): running prose, then bold text, code text and the lede.
- **Muted** (muted): labels, captions, table headers, command lines in the ladder, footer text.
- **The Bench** (bench-pink to bench-grey to bench-cool, at 135deg): the landing hero only. It is layered under a radial pink wash at the top left and an 18px dot grid (rgba(27,28,32,.09) dots).
- **Dark set** (dark-canvas, dark-ground, dark-ink, dark-body, dark-muted): the canvas, the far end of the bench, headings, prose and labels in dark mode. Soft and card greys in dark mode are color-mix steps of dark-canvas toward dark-ink (94% for soft). Hairlines are white at .12 and .22 alpha.

### Named Rules
**The Two Crimsons Rule.** Crimson appears in exactly two places: the TRY IT ONLINE pill and the reading light behind the wordmark. Links, hovers, focus rings and selection are ink. If a third crimson thing appears, it is a mistake.

**The Colour Means Tier Rule.** Green, olive, amber and the abstain grey are used only for tier marks and tier verdicts. They never decorate.

**The Neon Glows, Never Draws Rule.** On white, the lamp neons appear only as a glow. They are never used as a stroke, a border or text. In dark mode they become the stroke.

## Typography

**Display Font:** Saira Condensed (with Arial Narrow)
**Body Font:** Public Sans (with the system UI stack)
**Label/Mono Font:** JetBrains Mono (with ui-monospace)

**Character:** Condensed caps set the tone of a lab label: tall and tracked, always weight 400. Public Sans stays neutral so the chemistry can speak. Mono is how the site says "you type this" or "compare this".

### Hierarchy
- **Display** (400, clamp 2.2 to 3.3rem, 1.02, tracked .16rem, uppercase): the landing promise only. On phones it is 2.25rem.
- **Headline** (400, clamp 2 to 3rem, 1.1, tracked .1875rem, uppercase): page h1. When the h1 is a code identifier it drops to clamp 1.4 to 2.2rem with no tracking, the code keeps its case, and the chip is removed.
- **Title** (400, clamp 1.4 to 1.75rem, 1.2, tracked .125rem, uppercase): h2, with 2.4em above it.
- **Subtitle** (400, 1.25rem, 1.3, tracked .09375rem, uppercase): h3. Door-card titles use the same voice at 1.5rem.
- **Body** (400, 1rem, 1.7): prose, capped at 72ch. The landing lede is 1.02rem at 1.65 in body-strong, capped at 36rem. Bold text is weight 600.
- **Code** (400, .86em): inline chips. Blocks are .84rem at 1.65. Tabular numerals apply to code and tables.
- **Label** (400, .6875rem, tracked .125rem, uppercase, muted): table headers, sidebar captions, "On this page", field-list terms, rubrics, round-trip step labels, the install label, path-group headings.
- **Nav** (400, .75rem, tracked .125rem, uppercase): header section links, GitHub, the pill text and the tier names on the ladder (tracked .1875rem there).

### Named Rules
**The Three Jobs Rule.** Saira means a heading, Public Sans means prose, and JetBrains Mono means something typed or compared. A chemical name is prose, so it is set in Public Sans and never in mono.

**The Caps Are Light Rule.** Display caps are weight 400 and tracked. Never bold them.

**The Label Names A Thing Rule.** The mono caps label names a column, a group, a field or a step. It is never an eyebrow stacked over a headline.

## Layout

Documentation pages use the Shibuya three-column shell: the left global table of contents, the content column, and the right "On this page" rail. Below 1280px the right rail is capped at 16rem. The navbar is 60px tall with a hairline below it (a 1px shadow). The landing page widens the content to 76rem and sets one bench hero (padding clamp 1.25 to 3rem). From 1024px up the hero is two columns (1fr and 1.08fr, 2.5rem gap): the promise and doors on the left, the ladder on the right. Below that it stacks. The door cards pair up from 640px. The "Find your way" groups use auto-fit columns with a minimum of 11.5rem (1.5rem by 2rem gaps). Spacing follows a loose rhythm of about .9rem inside cards, 1.1rem card padding, 1.5rem grid gaps and 2.5rem between hero columns. Below 640px the hero drops to 1.25rem padding with a 24px radius, and the round trip collapses to one column. Tables are block elements with horizontal scroll below 900px, so the page itself never scrolls sideways.

## Elevation & Depth

This is a hybrid of a flat reading surface and one lifted layer. Pages are flat, and structure comes from hairlines and soft-grey wells. Only white cards that sit on the bench (door cards and the ladder) carry a shadow. It is ambient and two-layer, never offset. The TRY IT ONLINE pill has its own small, crimson-tinted shadow with an inset highlight.

### Shadow Vocabulary
- **Card lift** (`box-shadow: 0 1px 2px rgba(17,18,20,.05), 0 12px 30px -14px rgba(17,18,20,.14)`; dark: `0 1px 2px rgba(0,0,0,.4), 0 12px 30px -14px rgba(0,0,0,.6)`): white cards on the bench.
- **Pill** (`box-shadow: inset 0 1px 0 rgba(255,255,255,.28), 0 1px 2px rgba(157,24,48,.3), 0 6px 16px -8px rgba(157,24,48,.55)`): TRY IT ONLINE only. On hover it deepens to `0 10px 22px -10px rgba(157,24,48,.7)`.
- **Header seam** (`box-shadow: 0 1px 0 var(--ot-hair)`): the navbar underline.
- **PIN glow** (`filter: drop-shadow(0 0 5px` glow-pin at 80%`)`): the lit PIN mark only.

### Named Rules
**The Lift Needs A Bench Rule.** A shadow means "a card resting on the bench". Cards on the plain canvas (the round trip, admonitions, code) stay flat with a hairline.

## Shapes

Corners get rounder as the surface gets bigger: 6px for inline code, 8px for the current nav item, 10px for chrome (search), 14px for wells, code blocks, API signatures and admonitions, 22px for cards, 36px for the bench, and a full pill for the one action. Borders are 1px hairlines. Admonitions deliberately have no coloured side bar: the left border is the same hairline as the rest. The API body hangs off a 1px left hairline, which counts as a seam and not an accent bar. The tier rules are the one place with heavier lines, and each has a meaning: 6px double for PIN, 2px dashed for fallback, 2.6px dotted in muted for best effort, 1.6px solid in hair-strong for abstain. Tier marks are 18-unit ring SVGs in currentColor: two rings and a dot for PIN, a dashed ring and a dot for fallback, a dotted ring for best effort, a thin ring for abstain. Arrows and checks are inline stroked SVG at 1.6 to 1.8 stroke with round caps.

## Components

### Buttons
The site has one button, and it is the only action in the header.
- **Shape:** full pill (9999px).
- **TRY IT ONLINE:** a vertical crimson gradient (crimson-light to crimson-deep) with white mono caps at .75rem, tracked .125rem, padding .5rem 1rem .5rem 1.1rem, and a trailing 14px arrow-out SVG.
- **Hover / Focus:** the gradient settles to crimson over crimson-deep and the shadow deepens, over .2s with cubic-bezier(.16,1,.3,1). The focus outline is crimson-deep. On phones the padding is .45rem .8rem and the tracking .08rem.

### Cards / Containers
- **Door card:** white, 22px, card lift, with a 1px transparent border that turns ink on hover as the card rises 2px and its arrow slides 4px right (.25s, the same curve). Inside are a Saira title at 1.5rem, a code line with its output wearing the PIN double rule, a muted .88rem note, and an arrow pinned to the bottom right.
- **Ladder card:** white, 22px, card lift. Rungs are separated by hairlines. Each rung has a head (mark, mono caps tier name, muted mono id pushed to the right), the name wearing its rule, the muted command with an italic common name, and one plain sentence. The opened rung adds the read-back well and the verdict.
- **Round trip:** white, 22px, 1px hairline, no shadow, clipped. Steps sit in a two-column grid (11rem label column) separated by hairlines. The last row is soft grey and holds the green "same" verdict.
- **Admonition:** white, 14px, hairline on all four sides, .9rem 1.1rem padding, with the title as a mono caps label. The internal-module note is a soft-grey card with no title.

### Inputs / Fields
- **Search:** Public Sans, 1px hairline, 10px, canvas background. Focus turns the border ink and removes the outline.

### Navigation
- **Header:** tracked Saira wordmark (1.35rem, .375rem) with a muted mono "Docs" sub-label and the reading light (a crimson radial that sweeps in once over 2.4s). Section links are mono caps .75rem in body colour, and on hover turn ink with an underline at .35em offset. On phones the sub-label hides and the tracking tightens to .25rem.
- **Sidebar:** captions are mono caps labels. Links are body colour at .92rem. The current page is ink at weight 600 on a soft 8px chip.
- **Prose links:** ink with a 1px hair-strong underline at .2em. On hover the underline turns ink and thickens to 2px.

### Tier mark and tier role (signature)
Inline tier references render as a ring mark (1.1em) followed by the mono tier id, or a mono lamp label, with .32em gap and no wrapping, coloured by tier. The shape carries the tier and the colour confirms it, so the mark still reads in greyscale.

### The name and its rule (signature)
A chemical name is set in Public Sans in ink with the tier rule beneath it (.3rem below). A label that is not a name (an abstain reason) is muted and wears the abstain rule.

### Read-back well
A soft 14px well containing muted .78rem terms over ink mono InChIKeys (.8rem, tracked .02em), so the two keys can be compared line by line.

### API signature
A soft 14px strip in mono .9rem. The function name is ink at weight 600, and the module prefix and property words are muted. The body hangs from a 1px left hairline.

### Code block
A soft 14px block with a 1px hairline in mono at .84rem and 1.65 line height. Prompts are muted and cannot be selected.

## Do's and Don'ts

### Do:
- **Do** keep crimson to the TRY IT ONLINE pill and the reading light.
- **Do** give every name on the page its tier mark and its rule (double, dashed, dotted or thin solid), and give every decline its reason code.
- **Do** set commands, ids, InChIKeys and API signatures in JetBrains Mono, and set chemical names in Public Sans.
- **Do** separate content with 1px hairlines (hair) and group it in soft-grey wells.
- **Do** keep display and headings in Saira Condensed at weight 400, uppercase and tracked.
- **Do** hold prose to 72ch and let tables scroll inside themselves below 900px.
- **Do** use the card-lift shadow only for white cards resting on the bench.
- **Do** switch tier marks to the lamp neons in dark mode, and turn the green verdict text to glow-pin.
- **Do** stop the reading-light sweep, the PIN breaths and card transitions under prefers-reduced-motion.

### Don't:
- **Don't** use crimson for links, hovers, focus rings, warnings or emphasis. Those are ink.
- **Don't** use the tier greens, olive or amber for anything that is not a tier.
- **Don't** draw a lamp neon as a line, border or text on white.
- **Don't** put a coloured side bar on admonitions or callouts.
- **Don't** bold Saira display caps.
- **Don't** set a mono caps label above a headline as an eyebrow.
- **Don't** repeat the bench on inner pages. One bench, behind the landing hero.
- **Don't** replace the ring marks with icon-font glyphs or emoji. Marks and arrows are inline stroked SVG.
