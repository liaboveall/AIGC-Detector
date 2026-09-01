# Model and data usage notice

The root MIT License applies to the original source code in this repository. It does
not relicense model weights, datasets, pretrained checkpoints, images, or other
third-party materials.

## Ensemble vNext checkpoint

`weights/aigc-detector-ensemble-vnext.pt` is published for research, education,
reproducibility, and hackathon evaluation. No commercial-use permission is granted by
this repository for the checkpoint. Users are responsible for reviewing the terms of
the pretrained backbones and every upstream dataset before redistribution, deployment,
or derivative-model use.

In particular, the documented training lineage includes sources with non-commercial
or incompletely specified terms, including CommunityForensics-Small, GenImage, and the
pinned MS-COCOAI/Defactify source. Dataset image bodies are not distributed here.

## Responsible use

The checkpoint emits uncertain model scores. It does not establish image provenance,
authorship, fraud, or misconduct. Do not use it as the sole basis for consequential
moderation, disciplinary, employment, legal, or financial decisions. Human review and
independent target-domain validation are required.

The checkpoint and reports are provided without warranty. See the model card and error
analysis for known technical limitations.
