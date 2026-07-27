from __future__ import annotations

from pathlib import Path

ROOT = Path('.')


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding='utf-8')


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding='utf-8')


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text and old not in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly one occurrence, found {count}')
    return text.replace(old, new, 1)


def insert_after_table_label(text: str, label: str, insertion: str) -> str:
    if '\\label{fig:implementation-fairlab-instantiation}' in text:
        return text
    token = f'\\label{{{label}}}'
    idx = text.find(token)
    if idx < 0:
        raise RuntimeError(f'missing table label {label}')
    end_idx = text.find('\\end{table}', idx)
    if end_idx < 0:
        raise RuntimeError(f'missing end table after {label}')
    end_idx += len('\\end{table}')
    return text[:end_idx] + '\n\n' + insertion.strip() + '\n' + text[end_idx:]


def patch_methodology() -> None:
    rel = 'docs/LatexThesis/sections/04_methodology.tex'
    s = read(rel)
    s = s.replace('FAIR-Lab', 'FAIRLab')

    heading_map = {
        '\\section{Overview of the FAIRLab methodology}': '\\section{Overview of the FAIRLab Methodology}',
        '\\section{Research design and experimental assumptions}': '\\section{Research Design and Experimental Assumptions}',
        '\\section{Dataset construction and source consolidation}': '\\section{Dataset Construction and Source Consolidation}',
        '\\section{Review documentation and semantic decision records}': '\\section{Review Documentation and Semantic Decision Records}',
        '\\section{Human-in-the-loop review and final dataset freezing}': '\\section{Human-in-the-Loop Review and Final Dataset Freezing}',
        '\\section{Data partitioning and OOD evaluation strategy}': '\\section{Data Partitioning and OOD Evaluation Strategy}',
        '\\section{Proxy-model evaluation framework}': '\\section{Proxy-Model Evaluation Framework}',
        '\\section{Perturbation selection rationale and operational relevance}': '\\section{Perturbation Selection Rationale and Operational Relevance}',
        '\\section{Adversarial evaluation methodology}': '\\section{Adversarial Evaluation Methodology}',
        '\\section{Anti-forensic evaluation methodology}': '\\section{Anti-Forensic Evaluation Methodology}',
        '\\section{Forensic evaluation bundle and blind-input design}': '\\section{Forensic Evaluation Bundle and Blind-Input Design}',
        '\\section{Black-box software evaluation protocol}': '\\section{Black-Box Software Evaluation Protocol}',
        '\\section{Evaluation metrics}': '\\section{Evaluation Metrics}',
        '\\section{Explainability approach}': '\\section{Explainability Approach}',
        '\\section{Traceability and reproducibility controls}': '\\section{Traceability and Reproducibility Controls}',
    }
    for old, new in heading_map.items():
        if old in s:
            s = s.replace(old, new, 1)

    old = r'''\draw[arrow] (binary.south) -- (perturb.north);

\draw[arrow] (ood.south) |- (bundle.east);
\draw[arrow] (perturb.south) |- (bundle.west);'''
    new = r'''\draw[arrow] (binary.south) -- (perturb.north);

% Direct path for the unmodified binary samples.
\draw[arrow] (binary.west) -- ++(-1.0,0) |- (bundle.west);

% Derived adversarial and anti-forensic artifacts.
\draw[arrow] (perturb.south) |- (bundle.west);

% Separate clean OOD branch.
\draw[arrow] (ood.south) |- (bundle.east);'''
    s = replace_once(s, old, new, label='methodology direct clean branch')
    write(rel, s)


def patch_implementation() -> None:
    rel = 'docs/LatexThesis/sections/05_implementation.tex'
    s = read(rel)

    s = replace_once(
        s,
        r'\texttt{FAIRLAB\_CELLEBRITE\_INSEYETS\_RUN\_01} \\',
        r'\codepath{FAIRLAB_CELLEBRITE_INSEYETS_RUN_01} \\',
        label='Cellebrite run identifier',
    )

    old_eq = r'''\begin{equation}
1\,000\ \text{clean binary inputs}
+
500\ \text{OOD inputs}
+
5\,000\ \text{adversarial artifacts}
+
5\,000\ \text{anti-forensic artifacts}
=
11\,500\ \text{inputs}.
\end{equation}'''
    new_eq = r'''\begin{equation}
\begin{aligned}
1\,000\ \text{clean binary inputs}
&+ 500\ \text{OOD inputs} \\
&+ 5\,000\ \text{adversarial artifacts} \\
&+ 5\,000\ \text{anti-forensic artifacts} \\
&= 11\,500\ \text{inputs}.
\end{aligned}
\end{equation}'''
    s = replace_once(s, old_eq, new_eq, label='bundle equation')

    figure = r'''
\begin{figure}[p]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    >=latex,
    font=\small,
    box/.style={
        rectangle,
        rounded corners,
        draw=black!75,
        thick,
        align=center,
        text width=6.8cm,
        minimum height=1.2cm
    },
    wide/.style={
        box,
        text width=10.6cm
    },
    arrow/.style={
        ->,
        thick,
        draw=black!75
    }
]

\node[wide, fill=gray!10] (source) at (0,0)
{\textbf{Heterogeneous source image pool}\\
Source images and initial provenance records};

\node[wide, fill=blue!8] (review) at (0,-1.9)
{\textbf{Human-in-the-loop validation}\\
Technical validation, semantic review, cleaning, and dataset freezing};

\node[wide, fill=green!10] (frozen) at (0,-3.8)
{\textbf{Final frozen dataset: 1\,500 images}\\
500 \texttt{weapon} + 500 \texttt{non\_weapon} + 500 clean \gls{ood}};

\node[box, fill=yellow!12] (binary) at (-4.2,-6.3)
{\textbf{Binary evaluation subset: 1\,000 images}\\
500 \texttt{weapon} + 500 \texttt{non\_weapon}};

\node[box, fill=yellow!12] (ood) at (4.2,-6.3)
{\textbf{Separate clean \gls{ood} branch: 500 images}\\
No adversarial or anti-forensic generation};

\node[box, fill=orange!12] (perturb) at (-4.2,-8.8)
{\textbf{Perturbation generation from the binary subset}\\
5\,000 adversarial + 5\,000 anti-forensic artifacts};

\node[wide, fill=blue!8] (bundle) at (0,-11.4)
{\textbf{Forensic evaluation bundle: 11\,500 items}\\
1\,000 clean binary + 500 clean \gls{ood} + 5\,000 adversarial
+ 5\,000 anti-forensic};

\node[box, fill=orange!10] (proxy) at (-4.2,-14.1)
{\textbf{Transparent proxy-model evaluation}\\
EfficientNet-B0, ResNet18, and CLIP-based proxy};

\node[box, fill=orange!10] (blackbox) at (4.2,-14.1)
{\textbf{Black-box software evaluation}\\
Magnet AXIOM/Magnet.AI; Magnet Griffeye/T3K CORE;\\
Excire Foto 2025; Cellebrite Inseyets};

\node[box, fill=green!10] (proxout) at (-4.2,-16.8)
{\textbf{Proxy outputs}\\
Predictions, metrics, directional errors, confidence-related indicators,
and five-case Integrated Gradients analysis};

\node[box, fill=green!10] (norm) at (4.2,-16.8)
{\textbf{Black-box normalization}\\
Observable exports matched to hidden control metadata and converted into
documented operational decisions};

\node[wide, fill=gray!10] (analysis) at (0,-19.5)
{\textbf{Operational robustness analysis and traceability}\\
Normalized metrics, configuration-sensitive interpretation, provenance,
integrity controls, audit records, and human-review implications};

\draw[arrow] (source) -- (review);
\draw[arrow] (review) -- (frozen);
\draw[arrow] (frozen.south west) -- (binary.north);
\draw[arrow] (frozen.south east) -- (ood.north);
\draw[arrow] (binary) -- (perturb);

% Clean binary inputs enter the bundle directly.
\draw[arrow] (binary.west) -- ++(-1.0,0) |- (bundle.west);

% Perturbed binary artifacts enter the bundle.
\draw[arrow] (perturb) |- (bundle.west);

% The clean OOD branch enters the bundle without perturbation.
\draw[arrow] (ood) |- (bundle.east);

\draw[arrow] (bundle.south west) -- (proxy.north);
\draw[arrow] (bundle.south east) -- (blackbox.north);
\draw[arrow] (proxy) -- (proxout);
\draw[arrow] (blackbox) -- (norm);
\draw[arrow] (proxout.south east) -- (analysis.north);
\draw[arrow] (norm.south west) -- (analysis.north);

\end{tikzpicture}%
}
\caption[Concrete FAIRLab instantiation]{Concrete FAIRLab instantiation used
in this thesis. The 1\,500-image frozen dataset is divided into a 1\,000-image
binary branch and a separate 500-image clean \gls{ood} branch. Adversarial and
anti-forensic artifacts are generated only from the binary subset. The resulting
11\,500-item bundle supports parallel proxy-model and black-box software
evaluation, followed by output normalization, metrics, qualitative
explainability for EfficientNet-B0, and traceability controls}
\label{fig:implementation-fairlab-instantiation}
\end{figure}
'''
    s = insert_after_table_label(s, 'tab:implementation-forensic-bundle-composition', figure)
    write(rel, s)


def patch_results() -> None:
    rel = 'docs/LatexThesis/sections/06_results.tex'
    s = read(rel)

    s = replace_once(
        s,
        r'''\textit{Conf.} indicates
the mean confidence of the predictions.''',
        r'''\textit{Conf.} indicates the mean maximum predicted-class probability and is
reported only as an intra-model diagnostic indicator.''',
        label='clean confidence note',
    )

    s = replace_once(s, r'\textbf{Tot.} &', r'\textbf{Total} &', label='OOD total header')
    s = replace_once(
        s,
        '\\textbf{Pred. w} &\n\\textbf{Pred. nw} &',
        '\\textbf{\\shortstack{Weapon\\\\predictions}} &\n\\textbf{\\shortstack{Non-weapon\\\\predictions}} &',
        label='OOD prediction headers',
    )
    old_note = r'''\textit{Note.} \textit{Pred. w} indicates \texttt{weapon} predictions;
\textit{Pred. nw} indicates \texttt{non\_weapon} predictions;
\textit{W-rate} indicates the proportion of \gls{ood} samples classified as
\texttt{weapon}; \textit{Conf.} indicates the mean maximum predicted-class
probability; \textit{HC-count}'''
    new_note = r'''\textit{Note.} \textit{W-rate} indicates the proportion of \gls{ood}
samples classified as \texttt{weapon}; \textit{Conf.} indicates the mean
maximum predicted-class probability; \textit{HC-count}'''
    s = replace_once(s, old_note, new_note, label='OOD table note')

    old_conf = r'''EfficientNet-B0 and ResNet18 show \gls{ood} weapon rates of 0.370 and 0.438,
respectively. However, their higher mean confidence on \gls{ood} inputs may make these
assignments more consequential when confidence is used as a criterion for
automatic prioritization.'''
    new_conf = r'''EfficientNet-B0 and ResNet18 show \gls{ood} weapon rates of 0.370 and 0.438,
respectively. Within each architecture, high maximum predicted-class
probabilities may make such assignments operationally consequential when the
same model's score is used for automatic prioritization. These values remain
model-specific diagnostic indicators and must not be compared across
architectures as calibrated probabilities.'''
    s = replace_once(s, old_conf, new_conf, label='comparative OOD confidence paragraph')

    s = replace_once(
        s,
        r'\textbf{Adv. acc./drop} &',
        r'\textbf{\shortstack{Adversarial\\accuracy/drop}} &',
        label='adversarial table header',
    )
    write(rel, s)


def patch_background() -> None:
    rel = 'docs/LatexThesis/sections/02_background.tex'
    s = read(rel)
    replacements = {
        r'\subsection{Adversarial and Adversarial-Style Perturbations Considered in the Experimental Protocol}':
            r'\subsection{Adversarial and Adversarial-Style Perturbation Families}',
        r'\subsection{Anti-Forensic Transformations Considered in the Experimental Protocol}':
            r'\subsection{Image-Level Anti-Forensic Transformation Families}',
    }
    for old, new in replacements.items():
        if old in s:
            s = s.replace(old, new, 1)
    write(rel, s)


def patch_appendix() -> None:
    rel = 'docs/LatexThesis/sections/08_appendix.tex'
    s = read(rel)
    sentence = r'''The directory names containing \texttt{chapter5} or
\texttt{chapter\_5} are retained as historical repository identifiers. They
refer to artifacts used by the current experimental-results chapter and do not
indicate the present chapter number.'''
    if sentence not in s:
        marker = '\\end{longtable}\n\n\n% ─────────────────────────────────────────────────────────────────────────────\n\\section{Dataset Traceability and Human-in-the-Loop Selection}'
        if marker not in s:
            raise RuntimeError('appendix insertion marker not found')
        replacement = '\\end{longtable}\n\n' + sentence + '\n\n\n% ─────────────────────────────────────────────────────────────────────────────\n\\section{Dataset Traceability and Human-in-the-Loop Selection}'
        s = s.replace(marker, replacement, 1)
    write(rel, s)


def patch_repository_docs() -> None:
    rel = 'README.md'
    s = read(rel)
    s = replace_once(
        s,
        '| XAI | Five Integrated Gradients case studies selected for Chapter 5 |',
        '| XAI | Five Integrated Gradients case studies selected for the thesis results chapter |',
        label='README XAI chapter reference',
    )
    write(rel, s)

    rel = 'docs/artifact/REPRODUCIBILITY.md'
    s = read(rel)
    s = replace_once(
        s,
        '| 20 | `results/scripts/20_generate_experimental_reporting_assets.py` | Chapter 5 reporting |',
        '| 20 | `results/scripts/20_generate_experimental_reporting_assets.py` | Thesis reporting assets for the experimental-results chapter |',
        label='reproducibility step 20 chapter reference',
    )
    write(rel, s)


def main() -> None:
    patch_methodology()
    patch_implementation()
    patch_results()
    patch_background()
    patch_appendix()
    patch_repository_docs()


if __name__ == '__main__':
    main()
