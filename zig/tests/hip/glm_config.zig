//! The checkpoint's config.json against GLM-5.3-Flash's shape (TF_GLM_CONFIG; skipped when unset).
test "TF_GLM_CONFIG parses to the whole GLM-5.3-Flash shape: EXL3 experts, 34 KDA and 11 MLA layers, one MTP layer" {
    try @import("glm").config_checkpoint.check();
}
