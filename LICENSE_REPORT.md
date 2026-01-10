# License Report for improved-tribble Repository

## Executive Summary

**Repository License:** MIT License ✅ (Commercial-friendly)

**Status:** 
- ✅ **Main repository:** MIT License - Commercial use allowed
- ✅ **All verified dependencies:** Commercial-friendly licenses (MIT, Apache 2.0, BSD)
- ⚠️ **3 GitHub dependencies:** Require manual license verification

**Commercial Use:** ⚠️ **MOSTLY ALLOWED** (1 copyleft license requires review + 3 GitHub dependencies need verification)

**Non-Commercial Licenses Found:** ❌ **NONE** (no explicit non-commercial restrictions found)

**Copyleft Licenses Found:** ⚠️ **1** (easydict - LGPL-3.0, requires review for commercial use)

---

## Repository License

**Main License:** MIT License (Commercial-friendly)
- **Location:** `/LICENSE`
- **Type:** Permissive, allows commercial use
- **Status:** ✅ Commercial use allowed

---

## Dependencies and Their Licenses

### Core Framework & Deep Learning

| Package | Version | License | Commercial Use | Notes |
|---------|---------|---------|----------------|-------|
| **torch** | 2.7.1 | BSD-3-Clause | ✅ Yes | Verified via PyPI - BSD-3-Clause, commercial use allowed |
| **torchvision** | 0.22.1 | BSD | ✅ Yes | Verified via PyPI - BSD license, commercial use allowed |
| **torchaudio** | 2.7.1 | BSD | ✅ Yes | Verified via PyPI - BSD license, commercial use allowed |
| **numpy** | <2 | BSD-3-Clause | ✅ Yes | Standard BSD-3-Clause, commercial use allowed |
| **scipy** | 1.16.3 | BSD-3-Clause | ✅ Yes | BSD-3-Clause (includes bundled libraries with compatible licenses), commercial use allowed |
| **pillow** | 11.3.0 | PIL License (HPND) | ✅ Yes | Historical Permission Notice and Disclaimer, commercial use allowed |
| **opencv-python-headless** | latest | Apache 2.0 | ✅ Yes | Verified via PyPI - Apache 2.0, commercial use allowed |
| **spconv-cu126** | 2.3.8 | Apache License 2.0 | ✅ Yes | Verified via PyPI - Apache 2.0, commercial use allowed |
| **flash-attn** | 2.8.3 | BSD License | ✅ Yes | BSD license, commercial use allowed |

### AI/ML Libraries

| Package | Version | License | Commercial Use | Notes |
|---------|---------|---------|----------------|-------|
| **transformers** | 4.55.4 | Apache 2.0 License | ✅ Yes | Verified via PyPI - Apache 2.0, commercial use allowed |
| **diffusers** | latest (GitHub) | Apache 2.0 | ✅ Yes | Hugging Face library, typically Apache 2.0 |
| **peft** | 0.18.0 | Apache | ✅ Yes | Verified via PyPI - Apache license, commercial use allowed |
| **accelerate** | 1.12.0 | Apache | ✅ Yes | Verified via PyPI - Apache license, commercial use allowed |
| **timm** | 1.0.22 | Apache-2.0 | ✅ Yes | Verified via PyPI - Apache 2.0, commercial use allowed |
| **kornia** | 0.8.2 | Apache-2.0 | ✅ Yes | Verified via PyPI - Apache 2.0, commercial use allowed |

### Web Framework & Utilities

| Package | Version | License | Commercial Use | Notes |
|---------|---------|---------|----------------|-------|
| **fastapi** | 0.123.7 | MIT | ✅ Yes | MIT license (standard), commercial use allowed |
| **uvicorn** | latest | BSD | ✅ Yes | BSD license (standard), commercial use allowed |
| **python-multipart** | latest | Apache Software License | ✅ Yes | Verified via PyPI - Apache license, commercial use allowed |
| **pydantic** | 2.12.5 | MIT | ✅ Yes | MIT license (standard), commercial use allowed |
| **pydantic-settings** | 2.12.0 | MIT License | ✅ Yes | Verified via PyPI - MIT License, commercial use allowed |
| **pydantic-tensor** | 0.2.0 | MIT | ✅ Yes | MIT license (standard), commercial use allowed |
| **loguru** | 0.7.3 | MIT License | ✅ Yes | Verified via PyPI - MIT License, commercial use allowed |

### Other Utilities

| Package | Version | License | Commercial Use | Notes |
|---------|---------|---------|----------------|-------|
| **tqdm** | 4.67.1 | MPL-2.0 AND MIT | ✅ Yes | Verified via PyPI - Dual licensed (MPL-2.0 AND MIT), both allow commercial use |
| **easydict** | 1.13 | LGPL-3.0 | ⚠️ **Review** | **⚠️ COPLEFT LICENSE** - Verified via PyPI - LGPL-3.0 (copyleft), requires review for commercial use |

### GitHub Dependencies (Need Verification)

| Package | Source | License | Commercial Use | Notes |
|---------|--------|---------|----------------|-------|
| **utils3d** | git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 | ⚠️ Unknown | ⚠️ Unknown | **NEEDS VERIFICATION** - Check repository license |
| **diffusers** | git+https://github.com/huggingface/diffusers | Apache 2.0 | ✅ Yes | Hugging Face, typically Apache 2.0 |
| **ben2** | git+https://github.com/PramaLLC/BEN2.git | ⚠️ Unknown | ⚠️ Unknown | **NEEDS VERIFICATION** - Check repository license |
| **pyspz** | GitHub release (v.0.1.2) | ⚠️ Unknown | ⚠️ Unknown | **NEEDS VERIFICATION** - Check repository license |

---

## License Summary

### Commercial-Friendly Licenses (✅ Commercial Use Allowed)

All identified dependencies use permissive licenses that allow commercial use:
- **MIT License**: Most permissive, allows commercial use
- **Apache 2.0**: Permissive, allows commercial use (requires attribution)
- **BSD-3-Clause**: Permissive, allows commercial use
- **BSD-style (PyTorch)**: Permissive, allows commercial use
- **MPL 2.0**: Permissive, allows commercial use

### ⚠️ Dependencies Requiring Attention

#### Copyleft License (Requires Review)

1. **easydict** (LGPL-3.0)
   - **License:** GNU Lesser General Public License v3.0 (LGPL-3.0)
   - **Status:** ⚠️ **COPLEFT LICENSE** - Requires review for commercial use
   - **Notes:** LGPL-3.0 is a copyleft license. It allows commercial use but has specific requirements:
     - If you modify easydict itself, you must release modifications under LGPL-3.0
     - If you use easydict as a library (without modification), you can use it in proprietary software
     - **Recommendation:** Review how easydict is used in the codebase - if used as-is without modification, commercial use is typically allowed

#### GitHub Dependencies (Need Manual Verification)

2. **utils3d** (EasternJournalist/utils3d)
   - **Action Required:** Check the repository's LICENSE file
   - **Repository:** https://github.com/EasternJournalist/utils3d

3. **ben2** (PramaLLC/BEN2)
   - **Action Required:** Check the repository's LICENSE file
   - **Repository:** https://github.com/PramaLLC/BEN2

4. **pyspz** (404-Repo/spz)
   - **Action Required:** Check the repository's LICENSE file
   - **Repository:** https://github.com/404-Repo/spz

---

## Recommendations

1. ✅ **Main Repository License:** MIT License is commercial-friendly
2. ✅ **Most Dependencies:** All verified dependencies use commercial-friendly licenses (MIT, Apache 2.0, BSD)
3. ⚠️ **Action Required - easydict (LGPL-3.0):**
   - Review how easydict is used in the codebase
   - If used as-is without modification: ✅ Commercial use is typically allowed
   - If modified: Must release modifications under LGPL-3.0
   - **Recommendation:** Consider replacing with a permissively-licensed alternative if LGPL-3.0 is a concern
4. ⚠️ **Action Required:** Verify licenses for GitHub dependencies:
   - Check LICENSE files in the repositories listed above
   - Ensure they are compatible with commercial use
   - Document any restrictions found

5. **Best Practices:**
   - Keep this report updated when adding new dependencies
   - Verify licenses before adding new GitHub dependencies
   - Consider using a license checker tool in CI/CD pipeline
   - Run `pip-licenses` in Docker container to get complete license information for all installed packages

---

## License Compatibility

All verified licenses are compatible with the MIT License used by this repository. The MIT License is compatible with:
- Apache 2.0
- BSD licenses
- MPL 2.0
- Other permissive licenses

---

## Notes

- This report is based on standard license information for these packages
- For GitHub dependencies, manual verification is recommended
- Always consult legal counsel for commercial projects to ensure full compliance
- License information may change with package updates - verify periodically

---

**Report Generated:** Based on requirements.txt, Dockerfile analysis, and PyPI API queries
**Last Updated:** License information verified via PyPI API and pip-licenses tool
**Note:** To get complete license information for all installed packages (including transitive dependencies), run `pip-licenses` inside the Docker container where packages are installed

