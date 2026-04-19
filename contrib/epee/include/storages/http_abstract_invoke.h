
// Copyright (c) 2006-2013, Andrey N. Sabelnikov, www.sabelnikov.net
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
// * Neither the name of the Andrey N. Sabelnikov nor the
// names of its contributors may be used to endorse or promote products
// derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER  BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

#pragma once
#include <atomic>
#include <boost/utility/string_ref.hpp>
#include <chrono>
#include <sstream>
#include <string>
#include <zlib.h>
#include "byte_slice.h"
#include "portable_storage_template_helper.h"
#include "net/http_base.h"
#include "net/http_server_handlers_map2.h"

namespace epee
{
  namespace net_utils
  {
    template<class t_request, class t_response, class t_transport>
    bool invoke_http_json(const boost::string_ref uri, const t_request& out_struct, t_response& result_struct, t_transport& transport, std::chrono::milliseconds timeout = std::chrono::seconds(15), const boost::string_ref method = "POST")
    {
      std::string req_param;
      if(!serialization::store_t_to_json(out_struct, req_param))
        return false;

      http::fields_list additional_params;
      additional_params.push_back(std::make_pair("Content-Type","application/json; charset=utf-8"));

      const http::http_response_info* pri = NULL;
      if(!transport.invoke(uri, method, req_param, timeout, std::addressof(pri), std::move(additional_params)))
      {
        LOG_PRINT_L1("Failed to invoke http request to  " << uri);
        return false;
      }

      if(!pri)
      {
        LOG_PRINT_L1("Failed to invoke http request to  " << uri << ", internal error (null response ptr)");
        return false;
      }

      if(pri->m_response_code != 200)
      {
        LOG_PRINT_L1("Failed to invoke http request to  " << uri << ", wrong response code: " << pri->m_response_code);
        return false;
      }

      return serialization::load_t_from_json(result_struct, pri->m_body);
    }



    // Global counter used to generate unique per-session request IDs.
    // Seeded with a random component so IDs from different wallet runs don't collide in logs.
    inline uint64_t next_perf_request_id()
    {
      static std::atomic<uint64_t> seed{
        static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()) & 0xFFFFFFFFULL
      };
      static std::atomic<uint64_t> counter{0};
      const uint64_t n = counter.fetch_add(1, std::memory_order_relaxed);
      return (seed.load(std::memory_order_relaxed) << 20) | (n & 0xFFFFFULL);
    }

    template<class t_request, class t_response, class t_transport>
    bool invoke_http_bin(const boost::string_ref uri, const t_request& out_struct, t_response& result_struct, t_transport& transport, std::chrono::milliseconds timeout = std::chrono::seconds(15), const boost::string_ref method = "POST")
    {
      const uint64_t perf_req_id = next_perf_request_id();
      auto t_serial0 = std::chrono::steady_clock::now();
      byte_slice req_param;
      if(!serialization::store_t_to_binary(out_struct, req_param, 16 * 1024))
        return false;
      auto t_serial1 = std::chrono::steady_clock::now();
      const size_t req_size = req_param.size();

      http::fields_list additional_params;
      {
        std::ostringstream oss; oss << perf_req_id;
        additional_params.emplace_back("X-Perf-Req-Id", oss.str());
      }

      // Wall-clock timestamp (UNIX ms) for correlation with server logs (no shared monotonic clock).
      const auto t_send_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

      const http::http_response_info* pri = NULL;
      auto t_http0 = std::chrono::steady_clock::now();
      if(!transport.invoke(uri, method, boost::string_ref{reinterpret_cast<const char*>(req_param.data()), req_param.size()}, timeout, std::addressof(pri), std::move(additional_params)))
      {
        LOG_PRINT_L1("Failed to invoke http request to  " << uri);
        return false;
      }
      auto t_http1 = std::chrono::steady_clock::now();
      const auto t_recv_epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

      if(!pri)
      {
        LOG_PRINT_L1("Failed to invoke http request to  " << uri << ", internal error (null response ptr)");
        return false;
      }

      if(pri->m_response_code != 200)
      {
        LOG_PRINT_L1("Failed to invoke http request to  " << uri << ", wrong response code: " << pri->m_response_code);
        return false;
      }

      // Decompress gzip response if server sent Content-Encoding: gzip
      std::string decompressed_body;
      const std::string* body_ptr = &pri->m_body;
      const size_t raw_size = pri->m_body.size();
      bool was_gzip = false;
      auto t_decomp0 = std::chrono::steady_clock::now();
      if (pri->m_header_info.m_content_encoding.find("gzip") != std::string::npos)
      {
        was_gzip = true;
        z_stream zs{};
        if (inflateInit2(&zs, 15 + 16) != Z_OK) // 15+16 = gzip format
        {
          LOG_PRINT_L1("Failed to init gzip decompression");
          return false;
        }
        zs.next_in = (Bytef*)pri->m_body.data();
        zs.avail_in = pri->m_body.size();
        decompressed_body.resize(pri->m_body.size() * 4); // estimate 4x ratio
        int ret;
        do {
          zs.next_out = (Bytef*)decompressed_body.data() + zs.total_out;
          zs.avail_out = decompressed_body.size() - zs.total_out;
          ret = inflate(&zs, Z_NO_FLUSH);
          if (ret == Z_BUF_ERROR || (ret == Z_OK && zs.avail_out == 0))
            decompressed_body.resize(decompressed_body.size() * 2);
        } while (ret == Z_OK || ret == Z_BUF_ERROR);
        decompressed_body.resize(zs.total_out);
        inflateEnd(&zs);
        if (ret != Z_STREAM_END)
        {
          LOG_PRINT_L1("gzip decompression failed: " << ret);
          return false;
        }
        body_ptr = &decompressed_body;
      }
      auto t_decomp1 = std::chrono::steady_clock::now();

      // Limits raised from 65536*3 to 65536*16 to accommodate bulk block responses
      // (2000 blocks × ~40 txs × multiple nested structs easily exceeds 200k objects).
      static const constexpr epee::serialization::portable_storage::limits_t default_http_bin_limits = {
        65536 * 16, // objects  (~1M)
        65536 * 16, // fields   (~1M)
        65536 * 16, // strings  (~1M)
      };
      auto t_deser0 = std::chrono::steady_clock::now();
      bool ok = serialization::load_t_from_binary(result_struct, epee::strspan<uint8_t>(*body_ptr), &default_http_bin_limits);
      auto t_deser1 = std::chrono::steady_clock::now();

      // Compact PERF log covering entire HTTP cycle. Request-id + epoch timestamps allow
      // cross-machine correlation with the server's [PERF RPC] logs. Explicit "global"
      // category because wallet2.cpp's translation unit defines MONERO_DEFAULT_LOG_CATEGORY
      // as "wallet.wallet2" but this header may be included from other units where the
      // category is filtered (e.g. net.http:FATAL blocks everything http-ish).
      MCWARNING("global", "PERF invoke_http_bin id=" << perf_req_id
        << " uri=" << uri
        << " send_epoch_ms=" << t_send_epoch_ms
        << " recv_epoch_ms=" << t_recv_epoch_ms
        << " req_bytes=" << req_size
        << " http_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_http1 - t_http0).count()
        << " raw_resp_bytes=" << raw_size
        << " gzip=" << was_gzip
        << " decomp_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_decomp1 - t_decomp0).count()
        << " decomp_bytes=" << body_ptr->size()
        << " deser_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_deser1 - t_deser0).count()
        << " ser_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_serial1 - t_serial0).count()
        << " ok=" << ok);

      return ok;
    }

    template<class t_request, class t_response, class t_transport>
    bool invoke_http_json_rpc(const boost::string_ref uri, std::string method_name, const t_request& out_struct, t_response& result_struct, epee::json_rpc::error &error_struct, t_transport& transport, std::chrono::milliseconds timeout = std::chrono::seconds(15), const boost::string_ref http_method = "POST", const std::string& req_id = "0")
    {
      epee::json_rpc::request<t_request> req_t = AUTO_VAL_INIT(req_t);
      req_t.jsonrpc = "2.0";
      req_t.id = req_id;
      req_t.method = std::move(method_name);
      req_t.params = out_struct;
      epee::json_rpc::response<t_response, epee::json_rpc::error> resp_t = AUTO_VAL_INIT(resp_t);
      if(!epee::net_utils::invoke_http_json(uri, req_t, resp_t, transport, timeout, http_method))
      {
        error_struct = {};
        return false;
      }
      if(resp_t.error.code || resp_t.error.message.size())
      {
        error_struct = resp_t.error;
        LOG_ERROR("RPC call of \"" << req_t.method << "\" returned error: " << resp_t.error.code << ", message: " << resp_t.error.message);
        return false;
      }
      result_struct = resp_t.result;
      return true;
    }

    template<class t_request, class t_response, class t_transport>
    bool invoke_http_json_rpc(const boost::string_ref uri, std::string method_name, const t_request& out_struct, t_response& result_struct, t_transport& transport, std::chrono::milliseconds timeout = std::chrono::seconds(15), const boost::string_ref http_method = "POST", const std::string& req_id = "0")
    {
      epee::json_rpc::error error_struct;
      return invoke_http_json_rpc(uri, method_name, out_struct, result_struct, error_struct, transport, timeout, http_method, req_id);
    }

    template<class t_command, class t_transport>
    bool invoke_http_json_rpc(const boost::string_ref uri, typename t_command::request& out_struct, typename t_command::response& result_struct, t_transport& transport, std::chrono::milliseconds timeout = std::chrono::seconds(15), const boost::string_ref http_method = "POST", const std::string& req_id = "0")
    {
      return invoke_http_json_rpc(uri, t_command::methodname(), out_struct, result_struct, transport, timeout, http_method, req_id);
    }

  }
}
