# Provenance Update Change Summary

## Final Source Status

| Source bucket | Final status |
|---|---|
| `UCL SynthEye synthetic FAF` | Confirmed public dataset at dataset level |
| `derived_from_synthetic_faf` | Confirmed generated from held-out SynthEye synthetic FAF-like parents |
| `prepared_colour_fundus` | Confirmed public/prepared source bucket; exact upstream dataset unresolved |
| `prepared_oct_screenshot` | Confirmed public/prepared source bucket; exact upstream dataset unresolved |
| `prepared_cifar10_or_natural` | Confirmed public/prepared source bucket; exact upstream dataset unresolved |

## Per-Image Confirmation

No imported OOD row is per-image confirmed to an upstream public dataset. The
current package preserves benchmark path and SHA-256 hash, but not original
filename, original relative path, or source URL.

## Source-Bucket Confirmation

The 900 imported OOD images are traceable to public/prepared source buckets:

- 200 colour-fundus images
- 200 OCT-screenshot images
- 500 natural-image semantic outliers

## Unresolved Items

- Exact per-image APTOS/RFMiD/IRFundusSet/OLIVES/CIFAR-10/Open-Images mapping
  cannot be recovered from the current committed package.
- Imported OOD licence metadata remains source-bucket level.
- Future public release should preserve immutable per-image source and licence
  records at import time.

## Manuscript Sections Updated

- Code and Data Availability
- Section 4.2 image collection and data sources
- Table 3 parent/source descriptions
- Figure 8 caption
- Dataset integrity/provenance subsection
- Dataset limitations
- Conclusion

## Bibliography Entries Added

- `uclSyntheyeDataset2025`

No bibliography entry was added for APTOS, RFMiD, IRFundusSet, OLIVES,
CIFAR-10, or Open Images because the current committed benchmark cannot
confirm those datasets as exact upstream sources rather than preparation
candidates.

## PDF Status

- PDF: `latest_main.pdf`
- Compile log: `latest_main_compile.log`
- Page count: 83
- Compile status: Tectonic 0.16.9 exit code 0
- SHA-256: `047af790ec6843abf48264c2c742b53babbb961869ec2b8f62a3a7aae0c0fdae`
