---
layout: landing
---

```{raw} html
<div class="ot-home">
<section class="ot-hero" aria-labelledby="ot-promise">
  <div class="ot-hero-text">
    <h1 id="ot-promise">SMILES in.<br>A checked IUPAC name out.</h1>
    <p class="ot-lede">Orthonym builds the name from the rules of the IUPAC 2013 recommendations, aiming at the Preferred IUPAC Name. Then OPSIN, which never saw your structure, reads the name back, and the two structures are compared by InChIKey. Every name carries its tier. When no name passes, you get the reason instead of a guess.</p>
    <div class="ot-doors">
      <a class="ot-door" href="start/first-name.html">
        <span class="ot-door-title">Name a molecule</span>
        <span class="ot-door-code"><code>orthonym "CCO"</code><span class="ot-door-out">ethanol</span></span>
        <span class="ot-door-note">The first command, the five examples, and what <code>--provenance</code> prints.</span>
        <svg class="ot-door-arrow" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true" focusable="false"><path d="M4 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
      <a class="ot-door" href="reference/python-api.html">
        <span class="ot-door-title">Use it from code</span>
        <span class="ot-door-code"><code>from orthonym import name_compound</code></span>
        <span class="ot-door-note"><code>name_compound</code>, its parameters and the provenance fields, on one page.</span>
        <svg class="ot-door-arrow" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true" focusable="false"><path d="M4 12h15M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </a>
    </div>
    <div class="ot-install">
      <p class="ot-install-label">Install. Needs Python 3.10+ and a Java 11+ runtime.</p>
      <pre><code>pip install "git+https://github.com/Steinbeck-Lab/Orthonym.git"
orthonym --fetch-jars</code></pre>
      <p class="ot-install-note">No installing? <a href="https://orthonym.decimer.ai">Name a structure in the web app</a>.</p>
    </div>
  </div>

  <div class="ot-ladder" role="group" aria-labelledby="ot-ladder-title">
    <h2 id="ot-ladder-title" class="ot-sr-only">Examples of each tier</h2>

    <div class="ot-rung ot-rung--open">
      <div class="ot-rung-head">
        <svg class="ot-mark ot-mark--pin ot-lit" viewBox="0 0 18 18" width="26" height="26" aria-hidden="true" focusable="false"><circle fill="none" stroke="currentColor" cx="9" cy="9" r="8" stroke-width="1.25"/><circle fill="none" stroke="currentColor" cx="9" cy="9" r="5.5" stroke-width="1.25"/><circle cx="9" cy="9" r="3" fill="currentColor"/></svg>
        <span class="ot-rung-tier">PIN</span>
        <code class="ot-rung-id">pin_verified</code>
      </div>
      <p class="ot-name ot-rule--pin">1,3,7-trimethyl-3,7-dihydro-1<i>H</i>-purine-2,6-dione</p>
      <p class="ot-cmd"><code>orthonym "Cn1cnc2c1c(=O)n(C)c(=O)n2C"</code><span class="ot-cmd-note">caffeine</span></p>
      <dl class="ot-readback">
        <div><dt>InChIKey of your structure</dt><dd><code>RYYVLZVUVIJVGH-UHFFFAOYSA-N</code></dd></div>
        <div><dt>InChIKey of what OPSIN read back</dt><dd><code>RYYVLZVUVIJVGH-UHFFFAOYSA-N</code></dd></div>
      </dl>
      <p class="ot-verdict"><svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><path d="M3 8.5 6.5 12 13 4.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>Same molecule: a verified Preferred IUPAC Name.</p>
    </div>

    <div class="ot-rung">
      <div class="ot-rung-head">
        <svg class="ot-mark ot-mark--fallback" viewBox="0 0 18 18" width="26" height="26" aria-hidden="true" focusable="false"><circle fill="none" stroke="currentColor" cx="9" cy="9" r="7" stroke-width="1.75" stroke-dasharray="3.23 2.27"/><circle cx="9" cy="9" r="3" fill="currentColor"/></svg>
        <span class="ot-rung-tier">Fallback</span>
        <code class="ot-rung-id">pin_unverified</code>
      </div>
      <p class="ot-name ot-rule--fallback">(2<i>S</i>,3<i>R</i>,4<i>E</i>)-2-aminooctadec-4-ene-1,3-diol</p>
      <p class="ot-cmd"><code>orthonym --emit-tier valid "CCCCCCCCCCCCC/C=C/[C@H]([C@H](CO)N)O"</code><span class="ot-cmd-note">sphingosine</span></p>
      <p class="ot-rung-say">Read back by OPSIN to the same molecule; its preferred status is not certified.</p>
    </div>

    <div class="ot-rung">
      <div class="ot-rung-head">
        <svg class="ot-mark ot-mark--best-effort" viewBox="0 0 18 18" width="26" height="26" aria-hidden="true" focusable="false"><circle fill="none" stroke="currentColor" cx="9" cy="9" r="7" stroke-width="1.75" stroke-dasharray="1.1 3.3"/></svg>
        <span class="ot-rung-tier">Best effort</span>
        <code class="ot-rung-id">best_effort</code>
      </div>
      <p class="ot-name ot-rule--best-effort"><i>cis</i>-bicyclo[4.4.0]decane</p>
      <p class="ot-cmd"><code>orthonym --emit-tier best-effort "C1CC[C@H]2CCCC[C@H]2C1"</code><span class="ot-cmd-note">cis-decalin</span></p>
      <p class="ot-rung-say">From a last-resort producer; OPSIN read it back to the same molecule.</p>
    </div>

    <div class="ot-rung">
      <div class="ot-rung-head">
        <svg class="ot-mark ot-mark--abstain" viewBox="0 0 18 18" width="26" height="26" aria-hidden="true" focusable="false"><circle fill="none" stroke="currentColor" cx="9" cy="9" r="7" stroke-width="1"/></svg>
        <span class="ot-rung-tier">No name</span>
        <code class="ot-rung-id">abstain</code>
      </div>
      <p class="ot-name ot-name--label ot-rule--abstain">inorganic compound (not supported)</p>
      <p class="ot-cmd"><code>orthonym "O=[U](=O)=O"</code><span class="ot-cmd-note">uranium trioxide</span></p>
      <p class="ot-rung-say">Declined, with the reason code <code>UNSUPPORTED_ELEMENT</code>.</p>
    </div>
    <p class="ot-ladder-foot">Every line is the engine's own output. <a href="tiers.html">What the tiers mean</a>.</p>
  </div>
</section>

<section class="ot-paths" aria-labelledby="ot-paths-title">
  <h2 id="ot-paths-title">Find your way</h2>
  <div class="ot-paths-grid">
    <nav aria-label="Start" class="ot-path">
      <h3>Start</h3>
      <ul><li><a href="start/install.html">Install</a></li><li><a href="start/first-name.html">Your first name</a></li></ul>
    </nav>
    <nav aria-label="Use" class="ot-path">
      <h3>Use</h3>
      <ul><li><a href="use/command-line.html">Command line</a></li><li><a href="use/python.html">Python</a></li><li><a href="use/batch.html">Batch files</a></li><li><a href="use/provenance.html">The provenance row</a></li><li><a href="use/browser.html">In the browser</a></li></ul>
    </nav>
    <nav aria-label="Understand" class="ot-path">
      <h3>Understand</h3>
      <ul><li><a href="checking.html">How every name is checked</a></li><li><a href="tiers.html">Output tiers</a></li><li><a href="declines.html">Declines</a></li><li><a href="accuracy.html">Accuracy</a></li><li><a href="how-it-works.html">How it works</a></li></ul>
    </nav>
    <nav aria-label="Reference" class="ot-path">
      <h3>Reference</h3>
      <ul><li><a href="reference/python-api.html">Python API</a></li><li><a href="reference/command-line.html">Command-line options</a></li><li><a href="reference/internals/index.html">Internals</a></li></ul>
    </nav>
    <nav aria-label="Project" class="ot-path">
      <h3>Project</h3>
      <ul><li><a href="project/contributing.html">Contributing</a></li><li><a href="project/changelog.html">Changelog</a></li><li><a href="project/cite.html">How to cite</a></li><li><a href="project/licence.html">Licence</a></li><li><a href="project/llms.html">llms.txt</a></li></ul>
    </nav>
  </div>
  <p class="ot-paper">A paper describing Orthonym is in preparation. Until it is published, please <a href="project/cite.html">cite the software</a>.</p>
</section>
</div>
```

```{toctree}
:hidden:
:caption: Start

start/install
start/first-name
```

```{toctree}
:hidden:
:caption: Use

use/command-line
use/python
use/batch
use/provenance
use/browser
```

```{toctree}
:hidden:
:caption: Understand

checking
tiers
declines
accuracy
how-it-works
```

```{toctree}
:hidden:
:caption: Reference

reference/python-api
reference/command-line
reference/internals/index
```

```{toctree}
:hidden:
:caption: Project

project/contributing
project/changelog
project/cite
project/licence
project/llms
```
