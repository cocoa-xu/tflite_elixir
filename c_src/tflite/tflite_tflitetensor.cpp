#include "tflite_tflitetensor.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "../nif_utils.hpp"
#include "../helper.h"

int _tflitetensor_name(ErlNifEnv *env, TfLiteTensor * tensor, ERL_NIF_TERM &out) {
    auto tensor_name_str = TfLiteTensorName(tensor);
    ERL_NIF_TERM tensor_name;
    unsigned char * ptr;
    size_t len = strlen(tensor_name_str);
    if ((ptr = enif_make_new_binary(env, len, &tensor_name)) != nullptr) {
        strcpy((char *)ptr, tensor_name_str);
        out = tensor_name;
        return 0;
    } else {
        return 1;
    }
}

int _tflitetensor_shape(ErlNifEnv *env, TfLiteTensor * tensor, ERL_NIF_TERM &out) {
    size_t num_dims = TfLiteTensorNumDims(tensor);
    ERL_NIF_TERM * dims = (ERL_NIF_TERM *)enif_alloc(sizeof(ERL_NIF_TERM) * num_dims);
    if (dims == nullptr) {
        return 1;
    }
    for (size_t i = 0; i < num_dims; ++i) {
        size_t dim = TfLiteTensorDim(tensor, i);
        dims[i] = enif_make_uint64(env, dim);
    }
    out = enif_make_list_from_array(env, dims, (unsigned)num_dims);
    enif_free(dims);
    return 0;
}

int _tflitetensor_shape_signature(ErlNifEnv *env, TfLiteTensor * tensor, ERL_NIF_TERM &out) {
    if (tensor->dims_signature != nullptr && tensor->dims_signature->size != 0) {
        ERL_NIF_TERM shape_signature;
        if (erlang::nif::make_i64_list_from_c_array(env, tensor->dims_signature->size, (int64_t *)tensor->dims_signature, shape_signature)) {
            return 1;
        }
        out = shape_signature;
        return 0;
    } else {
        return _tflitetensor_shape(env, tensor, out);
    }
}

int _tflitetensor_type(ErlNifEnv *env, TfLiteTensor * tensor, ERL_NIF_TERM &out) {
    ERL_NIF_TERM tensor_type;
    if (tensor_type_to_erl_term(TfLiteTensorType(tensor), env, tensor_type)) {
        out = tensor_type;
        return 0;
    } else {
        return 1;
    }
}

int _tflitetensor_quantization_params(ErlNifEnv *env, TfLiteTensor * tensor, ERL_NIF_TERM &out) {
    const TfLiteQuantization quantization = tensor->quantization;
    float* scales_data = nullptr;
    int32_t* zero_points_data = nullptr;
    int32_t scales_size = 0;
    int32_t zero_points_size = 0;
    int32_t quantized_dimension = 0;

    if (quantization.type == kTfLiteAffineQuantization) {
        const TfLiteAffineQuantization* q_params =
                reinterpret_cast<const TfLiteAffineQuantization*>(quantization.params);
        if (q_params->scale) {
            scales_data = q_params->scale->data;
            scales_size = q_params->scale->size;
        }
        if (q_params->zero_point) {
            zero_points_data = q_params->zero_point->data;
            zero_points_size = q_params->zero_point->size;
        }
        quantized_dimension = q_params->quantized_dimension;
    }

    ERL_NIF_TERM scale;
    if (erlang::nif::make_f64_list_from_c_array(env, scales_size, scales_data, scale)) {
        return 1;
    }
    ERL_NIF_TERM zero_point;
    if (erlang::nif::make_i32_list_from_c_array(env, zero_points_size, zero_points_data, zero_point)) {
        return 1;
    }
    ERL_NIF_TERM quantized_dimension_term = enif_make_int(env, quantized_dimension);

    out = enif_make_tuple3(env, scale, zero_point, quantized_dimension_term);
    return 0;
}

int _tflitetensor_sparsity_params(ErlNifEnv *env, TfLiteTensor * tensor, ERL_NIF_TERM &out) {
    if (tensor->sparsity == nullptr) {
        out = enif_make_new_map(env);
        return 0;
    } else {
        auto param = tensor->sparsity;

        ERL_NIF_TERM sparsity_keys[3];
        ERL_NIF_TERM sparsity_vals[3];

        sparsity_keys[0] = erlang::nif::atom(env, "traversal_order");
        if (erlang::nif::make_i64_list_from_c_array(env, param->traversal_order->size, param->traversal_order->data, sparsity_vals[0])) {
            return 1;
        }

        sparsity_keys[1] = erlang::nif::atom(env, "block_map");
        if (erlang::nif::make_i64_list_from_c_array(env, param->block_map->size, param->block_map->data, sparsity_vals[1])) {
            return 1;
        }

        sparsity_keys[2] = erlang::nif::atom(env, "dim_metadata");
        ERL_NIF_TERM * dim_metadata = (ERL_NIF_TERM *)enif_alloc(sizeof(ERL_NIF_TERM) * param->dim_metadata_size);
        if (dim_metadata == nullptr) {
            return 1;
        }
        for (int i = 0; i < param->dim_metadata_size; i++) {
            ERL_NIF_TERM dim_metadata_i;
            ERL_NIF_TERM dim_metadata_i_keys[3];
            ERL_NIF_TERM dim_metadata_i_vals[3];

            if (param->dim_metadata[i].format == kTfLiteDimDense) {
                dim_metadata_i_keys[0] = erlang::nif::atom(env, "format");
                dim_metadata_i_vals[0] = erlang::nif::make(env, (long)0);

                dim_metadata_i_keys[1] = erlang::nif::atom(env, "dense_size");
                dim_metadata_i_vals[1] = erlang::nif::make(env, (long)param->dim_metadata[i].dense_size);

                enif_make_map_from_arrays(env, dim_metadata_i_keys, dim_metadata_i_vals, 2, &dim_metadata_i);
            } else {
                dim_metadata_i_keys[0] = erlang::nif::atom(env, "format");
                dim_metadata_i_vals[0] = erlang::nif::make(env, (long)1);

                const auto* array_segments = param->dim_metadata[i].array_segments;
                const auto* array_indices = param->dim_metadata[i].array_indices;

                dim_metadata_i_keys[1] = erlang::nif::atom(env, "array_segments");
                if (erlang::nif::make_i64_list_from_c_array(env, array_segments->size, array_segments->data, dim_metadata_i_vals[1])) {
                    return 1;
                }

                dim_metadata_i_keys[2] = erlang::nif::atom(env, "array_indices");
                if (erlang::nif::make_i64_list_from_c_array(env, array_indices->size, array_indices->data, dim_metadata_i_vals[2])) {
                    return 1;
                }

                enif_make_map_from_arrays(env, dim_metadata_i_keys, dim_metadata_i_vals, 3, &dim_metadata_i);
            }
            dim_metadata[i] = dim_metadata_i;
        }
        sparsity_vals[2] = enif_make_list_from_array(env, dim_metadata, (unsigned)param->dim_metadata_size);
        enif_free(dim_metadata);

        enif_make_map_from_arrays(env, sparsity_keys, sparsity_vals, 3, &out);

        return 0;
    }
}

ERL_NIF_TERM tflitetensor_type(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);

    ERL_NIF_TERM self_nif = argv[0];
    erlang_nif_res<TfLiteTensor *> *self_res;
    if (enif_get_resource(env, self_nif, erlang_nif_res<TfLiteTensor *>::type, (void **) &self_res)) {
        if (self_res->val) {
            ERL_NIF_TERM ret = erlang::nif::error(env, "invalid tensor");
            _tflitetensor_type(env, self_res->val, ret);
            return ret;
        } else {
            return erlang::nif::error(env, "oh nyo erlang");
        }
    } else {
        return erlang::nif::error(env, "cannot access resource");
    }
}

ERL_NIF_TERM tflitetensor_dims(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);

    ERL_NIF_TERM self_nif = argv[0];
    erlang_nif_res<TfLiteTensor *> *self_res;
    if (enif_get_resource(env, self_nif, erlang_nif_res<TfLiteTensor *>::type, (void **) &self_res)) {
        if (self_res->val) {
            ERL_NIF_TERM tensor_shape;
            if (_tflitetensor_shape(env, self_res->val, tensor_shape)) {
                return erlang::nif::error(env, "cannot allocate memory for storing tensor shape");
            }
            return erlang::nif::ok(env, tensor_shape);
        } else {
            return erlang::nif::error(env, "oh nyo erlang");
        }
    } else {
        return erlang::nif::error(env, "cannot access resource");
    }
}

ERL_NIF_TERM tflitetensor_quantization_params(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);

    ERL_NIF_TERM self_nif = argv[0];
    erlang_nif_res<TfLiteTensor *> *self_res;
    if (enif_get_resource(env, self_nif, erlang_nif_res<TfLiteTensor *>::type, (void **) &self_res)) {
        if (self_res->val) {
            ERL_NIF_TERM tensor_quantization_params;
            if (_tflitetensor_quantization_params(env, self_res->val, tensor_quantization_params)) {
                return erlang::nif::error(env, "cannot allocate memory for storing tensor quantization params");
            }
            return erlang::nif::ok(env, tensor_quantization_params);
        } else {
            return erlang::nif::error(env, "oh nyo erlang");
        }
    } else {
        return erlang::nif::error(env, "cannot access resource");
    }
}

ERL_NIF_TERM tflitetensor_to_binary(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);

    ERL_NIF_TERM self_nif = argv[0];
    erlang_nif_res<TfLiteTensor *> *self_res;
    if (enif_get_resource(env, self_nif, erlang_nif_res<TfLiteTensor *>::type, (void **) &self_res)) {
        if (self_res->val) {
            ErlNifBinary tensor_data;
            size_t tensor_size = self_res->val->bytes;
            if (!enif_alloc_binary(tensor_size, &tensor_data))
                return erlang::nif::error(env, "cannot allocate enough memory for the tensor");

            memcpy(tensor_data.data, self_res->val->data.raw, tensor_size);
            return erlang::nif::ok(env, enif_make_binary(env, &tensor_data));
        } else {
            return erlang::nif::error(env, "oh nyo erlang");
        }
    } else {
        return erlang::nif::error(env, "cannot access resource");
    }
}

ERL_NIF_TERM tflitetensor_set_data(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 2) return enif_make_badarg(env);

    ERL_NIF_TERM self_nif = argv[0];
    ERL_NIF_TERM data_nif = argv[1];
    ErlNifBinary data;
    erlang_nif_res<TfLiteTensor *> *self_res;
    if (enif_get_resource(env, self_nif, erlang_nif_res<TfLiteTensor *>::type, (void **) &self_res)) {
        if (self_res->val) {
            if (enif_inspect_binary(env, data_nif, &data)) {
                if (self_res->val->data.data == nullptr) {
                    return erlang::nif::error(env, "tensor is not allocated yet? Please call TFLite.Interpreter.allocateTensors first");
                } else {
                    memcpy(self_res->val->data.data, data.data, data.size);
                    return erlang::nif::ok(env);
                }
            } else {
                return erlang::nif::error(env, "cannot get input data");
            }
        } else {
            return erlang::nif::error(env, "oh nyo erlang");
        }
    } else {
        return erlang::nif::error(env, "cannot access resource");
    }
}
