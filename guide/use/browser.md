# Use it in the browser

The Orthonym web app, at [orthonym.decimer.ai](https://orthonym.decimer.ai), runs the same engine for chemists who do not want to install anything. Paste, upload or draw a structure, or send a whole file as one job and download the results.

The app is a separate project, [Orthonym-Web on GitHub](https://github.com/Steinbeck-Lab/Orthonym-Web).

## What it shows for each structure

Next to every result the app draws the structure OPSIN read back from the name, beside your input, so you can see the round trip yourself.

Each result lands on one of five states. The mark carries the state by its shape, so the five stay distinct in greyscale:

{lamp}`pin`
: The Preferred IUPAC Name. The engine's tier {tier}`pin_verified`: the strict PIN path built the name and verified it (OPSIN read it back to your structure; a name from the natural-product and metal-complex lists is matched to your exact structure instead).

{lamp}`fallback`
: A checked name whose preferred status is not certified. The engine's tiers {tier}`pin_unverified` and {tier}`systematic_verified`. Most of these names were read back by OPSIN to your structure; at the default tier a name that OPSIN read back with its constitution only, or could not read, lands here too. The app's own read-back verdict under the name says which.

{lamp}`best_effort`
: A name from a last-resort producer, not from the strict rules or the general engine. The engine's tier {tier}`best_effort`. The app prints its own read-back verdict under the name.

{lamp}`abstain`
: The engine declined rather than guess, so no name was made. The engine's tier {tier}`abstain`.

Error
: No name was made because RDKit could not read the input, or naming failed part-way, or a batch line ran out of time.

## When the app's own check did not run

When a result has no read-back from the app itself (for example with the read-back switched off), the app says "Name not checked here: no round-trip result" instead of the tier. When the app's read-back gives a different structure, it says "Round trip here gave a different structure".
