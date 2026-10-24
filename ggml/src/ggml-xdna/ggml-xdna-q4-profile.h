#pragma once

#include <cstring>

namespace ggml_xdna {

enum class q4_artifact_profile { canonical, prefetch_r2, prefetch_r2_gate_up };

struct q4_profile_artifacts {
    const char * name;
    const char * xclbin_sha256;
    const char * instruction_sha256[5]; // M=1024, 4096, 8192, 9216, Gate/Up M=9216
};

inline const q4_profile_artifacts * q4_artifacts(q4_artifact_profile profile) {
    static constexpr q4_profile_artifacts canonical = {
        "canonical",
        "9a743ff724463aa68d2dd62911c73eeeccad414788097ba90fd0bbff3b682d27",
        {"b9504aa178b3412d7adbdb4cdc641892acae90940095a5a6b1357021209a65b8",
         "9b89a2a63ee7c0c26273a6d25ac6511582769734b48bb9486b56281e1dd78c03",
         "e093f613fcb230dd12e0674a308d42dde248307e106dbcbc4ff20ba470ceb480",
         "d8e42a47f2b9e063a77edb513bd15ccc992204bd6e3cf28a519a821f97feee97",
         nullptr}
    };
    static constexpr q4_profile_artifacts prefetch_r2 = {
        "prefetch_r2",
        "c1865938f56ea6cf537cd983d38d32070466a319166d3cb66a3f3faa79857b8f",
        {"3420f1d0f84c8ab5ef648da43d1808a7f183cb7a0c793ccdc97b052bfc91adef",
         "9b9a258c15c18f7e84f344501b136f5e1f049f23726a349d29b11ff7f2bc88e7",
         "2acb5b2681947fdf029aa33089ff6323b457410748a68d6702d9eb272bf64563",
         "ed5eb57b3b0ed8823c1d64890c50de91a4f0f99a9a797e18a9454fc40b78cf44",
         nullptr}
    };
    static constexpr q4_profile_artifacts prefetch_r2_gate_up = {
        "prefetch_r2_gate_up",
        "6c8d80d654a981afeea2457d0974086c92b423c1b58a612dc506b62af3f15766",
        {"3420f1d0f84c8ab5ef648da43d1808a7f183cb7a0c793ccdc97b052bfc91adef",
         "9b9a258c15c18f7e84f344501b136f5e1f049f23726a349d29b11ff7f2bc88e7",
         "2acb5b2681947fdf029aa33089ff6323b457410748a68d6702d9eb272bf64563",
         "ed5eb57b3b0ed8823c1d64890c50de91a4f0f99a9a797e18a9454fc40b78cf44",
         "0eb1a652b48c4951a30804e975a0788081402b965db828a874969bbfd159705c"}
    };
    switch (profile) {
        case q4_artifact_profile::canonical: return &canonical;
        case q4_artifact_profile::prefetch_r2: return &prefetch_r2;
        case q4_artifact_profile::prefetch_r2_gate_up: return &prefetch_r2_gate_up;
    }
    return nullptr;
}

inline bool parse_q4_artifact_profile(const char * value, q4_artifact_profile * result) {
    if (result == nullptr) return false;
    if (value == nullptr || std::strcmp(value, "canonical") == 0) {
        *result = q4_artifact_profile::canonical;
        return true;
    }
    if (std::strcmp(value, "prefetch_r2") == 0) {
        *result = q4_artifact_profile::prefetch_r2;
        return true;
    }
    if (std::strcmp(value, "prefetch_r2_gate_up") == 0) {
        *result = q4_artifact_profile::prefetch_r2_gate_up;
        return true;
    }
    return false;
}

inline bool q4_profiles_compatible(q4_artifact_profile active, q4_artifact_profile requested) {
    return active == requested;
}
} // namespace ggml_xdna
