#include "axflow/model/model.h"
#include "model_internal.h"
#include "../device/device_internal.h"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace axflow {

Model::Model(Device& device, const AxflowConfig& cfg)
    : impl_(std::make_unique<Impl>())
{
    auto* dev = internal_impl(device);
    if (!dev || !dev->ctx || !dev->connection) {
        throw std::runtime_error("axflow::Model: device not connected");
    }
    impl_->ctx        = dev->ctx;
    impl_->connection = dev->connection;

    // resolve model.json path inside model_dir
    std::filesystem::path mdir(cfg.model_dir);
    std::filesystem::path mpath = mdir / "model.json";

    impl_->model = axr_load_model(impl_->ctx, mpath.string().c_str());
    if (!impl_->model) {
        std::string err = axr_last_error_string(AXR_OBJECT(impl_->ctx));
        throw std::runtime_error("axflow::Model: load_model failed: " + err);
    }

    // properties — single sub-device, num_cores from config
    std::string props_str = "input_dmabuf=0;num_sub_devices=1;aipu_cores="
                          + std::to_string(cfg.num_cores);
    auto* props = axr_create_properties(impl_->ctx, props_str.c_str());

    impl_->instance = axr_load_model_instance(impl_->connection, impl_->model, props);
    if (!impl_->instance) {
        std::string err = axr_last_error_string(AXR_OBJECT(impl_->ctx));
        throw std::runtime_error("axflow::Model: instance creation failed: " + err);
    }

    // cache tensor info + allocate buffers
    int n_in  = axr_num_model_inputs (impl_->model);
    int n_out = axr_num_model_outputs(impl_->model);

    impl_->input_infos.resize(n_in);
    impl_->input_bufs.resize(n_in);
    for (int i = 0; i < n_in; ++i) {
        impl_->input_infos[i] = axr_get_model_input(impl_->model, i);
        impl_->input_bufs[i].resize(axr_tensor_size(&impl_->input_infos[i]));
    }

    impl_->output_infos.resize(n_out);
    impl_->output_bufs.resize(n_out);
    for (int i = 0; i < n_out; ++i) {
        impl_->output_infos[i] = axr_get_model_output(impl_->model, i);
        impl_->output_bufs[i].resize(axr_tensor_size(&impl_->output_infos[i]));
    }
}

Model::~Model() {
    if (impl_ && impl_->instance) {
        axr_destroy(reinterpret_cast<const axrObject*>(impl_->instance));
    }
    if (impl_ && impl_->model) {
        axr_destroy(reinterpret_cast<const axrObject*>(impl_->model));
    }
}

Model::Model(Model&&) noexcept            = default;
Model& Model::operator=(Model&&) noexcept = default;

int Model::num_inputs () const { return impl_->input_infos.size();  }
int Model::num_outputs() const { return impl_->output_infos.size(); }

static std::vector<int> shape_of(const axrTensorInfo& t) {
    std::vector<int> s(t.ndims);
    for (int i = 0; i < t.ndims; ++i) s[i] = t.dims[i];
    return s;
}

std::vector<int> Model::input_shape (int i) const { return shape_of(impl_->input_infos.at(i));  }
std::vector<int> Model::output_shape(int i) const { return shape_of(impl_->output_infos.at(i)); }

float Model::input_scale     (int i) const { return impl_->input_infos.at(i).scale;       }
int   Model::input_zero_point(int i) const { return impl_->input_infos.at(i).zero_point;  }
float Model::output_scale    (int i) const { return impl_->output_infos.at(i).scale;      }
int   Model::output_zero_point(int i) const { return impl_->output_infos.at(i).zero_point; }

} // namespace axflow