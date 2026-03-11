$ErrorActionPreference = "Stop"

$root = "C:\Users\407407\Desktop\vllm\github\MQA_FastPath"
$targetDirs = @(
    (Join-Path $root "A&B"),
    (Join-Path $root "C&D"),
    (Join-Path $root "no_spec")
)

foreach ($dir in $targetDirs) {
    if (-not (Test-Path -LiteralPath $dir)) {
        throw "Directory not found: $dir"
    }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = Join-Path $root ("extracted_tables_" + $ts)
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

function Get-FileContext {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo]$File,
        [Parameter(Mandatory = $true)]
        [string[]]$Dirs
    )

    $group = ""
    foreach ($d in $Dirs) {
        if ($File.FullName.StartsWith($d, [System.StringComparison]::OrdinalIgnoreCase)) {
            $group = Split-Path -Leaf $d
            break
        }
    }

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($File.Name)
    $kind = "other"
    $suite = ""
    $k = $null

    if ($baseName -eq "summary") {
        $kind = "summary"
    } elseif ($baseName -match "^(?<kind>bench|key|meta|probe|server)_(?<suite>.+?)(?:_k(?<k>\d+))?$") {
        $kind = $matches["kind"]
        $suite = $matches["suite"]
        if ($matches["k"]) {
            $k = [int]$matches["k"]
        }
    }

    return [pscustomobject]@{
        data_group = $group
        kind       = $kind
        suite      = $suite
        k          = $k
    }
}

function Try-ParseDouble {
    param([string]$Text)
    if ($null -eq $Text) { return $null }
    $clean = $Text.Trim()
    if ($clean -eq "") { return $null }
    $clean = $clean -replace ",", ""
    $clean = $clean.TrimEnd("%")
    $clean = $clean.TrimEnd(".")
    $value = 0.0
    $ok = [double]::TryParse(
        $clean,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$value
    )
    if ($ok) { return $value }
    return $null
}

$allFiles = foreach ($dir in $targetDirs) {
    Get-ChildItem -LiteralPath $dir -Recurse -Force -File
}
$allFiles = $allFiles | Sort-Object FullName

$fileInventory = New-Object System.Collections.Generic.List[object]
$summaryRows = New-Object System.Collections.Generic.List[object]
$metaProbeKvRows = New-Object System.Collections.Generic.List[object]
$benchMetricsRows = New-Object System.Collections.Generic.List[object]
$logIssueRows = New-Object System.Collections.Generic.List[object]
$logKvRows = New-Object System.Collections.Generic.List[object]

$rawLinesPath = Join-Path $outputDir "all_log_lines.csv"
$rawHeaderWritten = $false

$benchLabelMap = @{
    "Successful requests"                            = "successful_requests"
    "Benchmark duration (s)"                         = "benchmark_duration_s"
    "Total input tokens"                             = "total_input_tokens"
    "Total generated tokens"                         = "total_generated_tokens"
    "Request throughput (req/s)"                     = "request_throughput_req_s"
    "Output token throughput (tok/s)"                = "output_token_throughput_tok_s"
    "Total Token throughput (tok/s)"                 = "total_token_throughput_tok_s"
    "Mean TTFT (ms)"                                 = "mean_ttft_ms"
    "Median TTFT (ms)"                               = "median_ttft_ms"
    "P99 TTFT (ms)"                                  = "p99_ttft_ms"
    "Mean TPOT (ms)"                                 = "mean_tpot_ms"
    "Median TPOT (ms)"                               = "median_tpot_ms"
    "P99 TPOT (ms)"                                  = "p99_tpot_ms"
    "Mean ITL (ms)"                                  = "mean_itl_ms"
    "Median ITL (ms)"                                = "median_itl_ms"
    "P99 ITL (ms)"                                   = "p99_itl_ms"
}

foreach ($file in $allFiles) {
    $ctx = Get-FileContext -File $file -Dirs $targetDirs
    $isText = @(".log", ".csv") -contains $file.Extension.ToLowerInvariant()
    $lineCount = $null
    if ($isText) {
        $lineCount = (Get-Content -LiteralPath $file.FullName | Measure-Object -Line).Lines
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash

    $fileInventory.Add([pscustomobject]@{
        data_group      = $ctx.data_group
        file_name       = $file.Name
        file_path       = $file.FullName
        extension       = $file.Extension
        source_kind     = $ctx.kind
        suite_from_name = $ctx.suite
        k_from_name     = $ctx.k
        size_bytes      = $file.Length
        line_count      = $lineCount
        last_write_time = $file.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        sha256          = $hash
    }) | Out-Null

    if ($file.Extension -ieq ".csv" -and $file.Name -ieq "summary.csv") {
        $rows = Import-Csv -LiteralPath $file.FullName
        $rowNo = 0
        foreach ($r in $rows) {
            $rowNo++
            $summaryRows.Add([pscustomobject]@{
                data_group                        = $ctx.data_group
                source_file                       = $file.FullName
                row_no                            = $rowNo
                suite                             = $r.suite
                k                                 = $r.k
                status                            = $r.status
                throughput_tps                    = $r.throughput_tps
                mean_tpot_ms                      = $r.mean_tpot_ms
                mean_itl_ms                       = $r.mean_itl_ms
                mqa_mode                          = $r.mqa_mode
                supports_gpu_multi_step           = $r.supports_gpu_multi_step
                fastpath_hit_rate                 = $r.fastpath_hit_rate
                scoring_time_ms                   = $r.scoring_time_ms
                verification_time_ms              = $r.verification_time_ms
                base_worker_or_runner_total_ms    = $r."base_worker_total_ms|base_runner_total_ms"
            }) | Out-Null
        }
    }

    if ($file.Extension -ieq ".log") {
        $lines = Get-Content -LiteralPath $file.FullName

        $warningCount = ($lines | Where-Object { $_ -match "\bWARNING\b" }).Count
        $errorCount = ($lines | Where-Object { $_ -match "\bERROR\b" }).Count
        $tracebackCount = ($lines | Where-Object { $_ -match "Traceback \(most recent call last\)" }).Count
        $serverFailCount = ($lines | Where-Object { $_ -match "SERVER_FAIL" }).Count

        $logIssueRows.Add([pscustomobject]@{
            data_group      = $ctx.data_group
            file_path       = $file.FullName
            source_kind     = $ctx.kind
            suite_from_name = $ctx.suite
            k_from_name     = $ctx.k
            warning_count   = $warningCount
            error_count     = $errorCount
            traceback_count = $tracebackCount
            server_fail_tag = $serverFailCount
        }) | Out-Null

        $rawRows = New-Object System.Collections.Generic.List[object]
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $rawRows.Add([pscustomobject]@{
                data_group  = $ctx.data_group
                file_path   = $file.FullName
                source_kind = $ctx.kind
                line_no     = $i + 1
                line_text   = $lines[$i]
            }) | Out-Null
        }
        if ($rawRows.Count -gt 0) {
            if (-not $rawHeaderWritten) {
                $rawRows | Export-Csv -LiteralPath $rawLinesPath -NoTypeInformation -Encoding UTF8
                $rawHeaderWritten = $true
            } else {
                $rawRows | Export-Csv -LiteralPath $rawLinesPath -NoTypeInformation -Encoding UTF8 -Append
            }
        }

        if ($ctx.kind -in @("meta", "probe")) {
            for ($i = 0; $i -lt $lines.Count; $i++) {
                $lineNo = $i + 1
                $line = $lines[$i]
                if ($line -match "^\[(META|PROBE)\]\s*(.+)$") {
                    $tag = $matches[1]
                    $payload = $matches[2]

                    if ($payload -match "^(serve_cmd|bench_cmd)=(.+)$") {
                        $metaProbeKvRows.Add([pscustomobject]@{
                            data_group  = $ctx.data_group
                            file_path   = $file.FullName
                            source_kind = $ctx.kind
                            suite       = $ctx.suite
                            k           = $ctx.k
                            tag         = $tag
                            line_no     = $lineNo
                            key         = $matches[1]
                            value       = $matches[2]
                        }) | Out-Null
                        continue
                    }

                    $eqCount = [regex]::Matches($payload, "[A-Za-z_][A-Za-z0-9_]*=").Count
                    if ($eqCount -eq 1 -and $payload -match "^([A-Za-z_][A-Za-z0-9_]*)=(.+)$") {
                        $metaProbeKvRows.Add([pscustomobject]@{
                            data_group  = $ctx.data_group
                            file_path   = $file.FullName
                            source_kind = $ctx.kind
                            suite       = $ctx.suite
                            k           = $ctx.k
                            tag         = $tag
                            line_no     = $lineNo
                            key         = $matches[1]
                            value       = $matches[2]
                        }) | Out-Null
                    } else {
                        $kvMatches = [regex]::Matches($payload, "([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")
                        foreach ($m in $kvMatches) {
                            $metaProbeKvRows.Add([pscustomobject]@{
                                data_group  = $ctx.data_group
                                file_path   = $file.FullName
                                source_kind = $ctx.kind
                                suite       = $ctx.suite
                                k           = $ctx.k
                                tag         = $tag
                                line_no     = $lineNo
                                key         = $m.Groups[1].Value
                                value       = $m.Groups[2].Value
                            }) | Out-Null
                        }
                    }
                }
            }
        }

        if ($ctx.kind -eq "bench") {
            $metrics = [ordered]@{
                data_group                     = $ctx.data_group
                source_file                    = $file.FullName
                suite                          = $ctx.suite
                k                              = $ctx.k
                successful_requests            = $null
                benchmark_duration_s           = $null
                total_input_tokens             = $null
                total_generated_tokens         = $null
                request_throughput_req_s       = $null
                output_token_throughput_tok_s  = $null
                total_token_throughput_tok_s   = $null
                mean_ttft_ms                   = $null
                median_ttft_ms                 = $null
                p99_ttft_ms                    = $null
                mean_tpot_ms                   = $null
                median_tpot_ms                 = $null
                p99_tpot_ms                    = $null
                mean_itl_ms                    = $null
                median_itl_ms                  = $null
                p99_itl_ms                     = $null
            }

            foreach ($line in $lines) {
                if ($line -match "^\s*([^:]+):\s+(.+?)\s*$") {
                    $label = $matches[1].Trim()
                    $val = $matches[2].Trim()
                    if ($benchLabelMap.ContainsKey($label)) {
                        $key = $benchLabelMap[$label]
                        $metrics[$key] = $val
                    }
                }
            }
            $benchMetricsRows.Add([pscustomobject]$metrics) | Out-Null
        }

        for ($i = 0; $i -lt $lines.Count; $i++) {
            $lineNo = $i + 1
            $line = $lines[$i]
            $kvMatches = [regex]::Matches($line, "([A-Za-z_][A-Za-z0-9_]*)=(\[[^\]]*\]|<[^>]*>|[^ ]+)")
            foreach ($m in $kvMatches) {
                $value = $m.Groups[2].Value.Trim().TrimEnd(",")
                $logKvRows.Add([pscustomobject]@{
                    data_group      = $ctx.data_group
                    file_path       = $file.FullName
                    source_kind     = $ctx.kind
                    suite_from_name = $ctx.suite
                    k_from_name     = $ctx.k
                    line_no         = $lineNo
                    key             = $m.Groups[1].Value
                    value           = $value
                }) | Out-Null
            }
        }
    }
}

$fileInventoryPath = Join-Path $outputDir "file_inventory.csv"
$summaryCombinedPath = Join-Path $outputDir "summary_combined.csv"
$metaProbeKvPath = Join-Path $outputDir "meta_probe_kv.csv"
$benchMetricsPath = Join-Path $outputDir "bench_metrics.csv"
$logIssuePath = Join-Path $outputDir "log_issue_stats.csv"
$logKvPath = Join-Path $outputDir "log_key_values.csv"
$logLastPath = Join-Path $outputDir "log_last_values.csv"
$logNumericPath = Join-Path $outputDir "log_numeric_stats.csv"
$masterRunPath = Join-Path $outputDir "master_run_table.csv"

$fileInventory | Sort-Object data_group, source_kind, file_name | Export-Csv -LiteralPath $fileInventoryPath -NoTypeInformation -Encoding UTF8
$summaryRows | Sort-Object data_group, suite, @{Expression = { [int]($_.k) }} | Export-Csv -LiteralPath $summaryCombinedPath -NoTypeInformation -Encoding UTF8
$metaProbeKvRows | Sort-Object data_group, file_path, line_no, key | Export-Csv -LiteralPath $metaProbeKvPath -NoTypeInformation -Encoding UTF8
$benchMetricsRows | Sort-Object data_group, suite, k | Export-Csv -LiteralPath $benchMetricsPath -NoTypeInformation -Encoding UTF8
$logIssueRows | Sort-Object data_group, file_path | Export-Csv -LiteralPath $logIssuePath -NoTypeInformation -Encoding UTF8
$logKvRows | Sort-Object data_group, file_path, line_no, key | Export-Csv -LiteralPath $logKvPath -NoTypeInformation -Encoding UTF8

$lastRows = New-Object System.Collections.Generic.List[object]
$groupedLast = $logKvRows | Group-Object file_path, key
foreach ($g in $groupedLast) {
    $sorted = $g.Group | Sort-Object line_no
    $last = $sorted[-1]
    $lastRows.Add([pscustomobject]@{
        data_group      = $last.data_group
        file_path       = $last.file_path
        source_kind     = $last.source_kind
        suite_from_name = $last.suite_from_name
        k_from_name     = $last.k_from_name
        key             = $last.key
        last_line_no    = $last.line_no
        last_value      = $last.value
    }) | Out-Null
}
$lastRows | Sort-Object data_group, file_path, key | Export-Csv -LiteralPath $logLastPath -NoTypeInformation -Encoding UTF8

$numericRows = New-Object System.Collections.Generic.List[object]
foreach ($row in $logKvRows) {
    $num = Try-ParseDouble -Text $row.value
    if ($null -ne $num) {
        $numericRows.Add([pscustomobject]@{
            data_group      = $row.data_group
            file_path       = $row.file_path
            source_kind     = $row.source_kind
            suite_from_name = $row.suite_from_name
            k_from_name     = $row.k_from_name
            key             = $row.key
            line_no         = $row.line_no
            value_num       = $num
        }) | Out-Null
    }
}

$numericStats = New-Object System.Collections.Generic.List[object]
$groupedNumeric = $numericRows | Group-Object file_path, key
foreach ($g in $groupedNumeric) {
    $rows = $g.Group
    $vals = $rows | ForEach-Object { $_.value_num }
    $sorted = $rows | Sort-Object line_no
    $last = $sorted[-1]
    $numericStats.Add([pscustomobject]@{
        data_group      = $last.data_group
        file_path       = $last.file_path
        source_kind     = $last.source_kind
        suite_from_name = $last.suite_from_name
        k_from_name     = $last.k_from_name
        key             = $last.key
        count           = $vals.Count
        min             = ($vals | Measure-Object -Minimum).Minimum
        max             = ($vals | Measure-Object -Maximum).Maximum
        avg             = [math]::Round((($vals | Measure-Object -Average).Average), 6)
        last_value      = $last.value_num
        last_line_no    = $last.line_no
    }) | Out-Null
}
$numericStats | Sort-Object data_group, file_path, key | Export-Csv -LiteralPath $logNumericPath -NoTypeInformation -Encoding UTF8

$benchIndex = @{}
foreach ($b in $benchMetricsRows) {
    $idx = "{0}|{1}|{2}" -f $b.data_group, $b.suite, $b.k
    $benchIndex[$idx] = $b
}

$issueIndex = @{}
foreach ($i in $logIssueRows | Where-Object { $_.source_kind -eq "server" }) {
    $idx = "{0}|{1}|{2}" -f $i.data_group, $i.suite_from_name, $i.k_from_name
    $issueIndex[$idx] = $i
}

$masterRows = New-Object System.Collections.Generic.List[object]
foreach ($s in $summaryRows) {
    $idx = "{0}|{1}|{2}" -f $s.data_group, $s.suite, $s.k
    $bench = $null
    if ($benchIndex.ContainsKey($idx)) { $bench = $benchIndex[$idx] }
    $issue = $null
    if ($issueIndex.ContainsKey($idx)) { $issue = $issueIndex[$idx] }

    $masterRows.Add([pscustomobject]@{
        data_group                      = $s.data_group
        suite                           = $s.suite
        k                               = $s.k
        status                          = $s.status
        mqa_mode                        = $s.mqa_mode
        supports_gpu_multi_step         = $s.supports_gpu_multi_step
        throughput_tps_summary          = $s.throughput_tps
        mean_tpot_ms_summary            = $s.mean_tpot_ms
        mean_itl_ms_summary             = $s.mean_itl_ms
        scoring_time_ms_summary         = $s.scoring_time_ms
        verification_time_ms_summary    = $s.verification_time_ms
        base_worker_or_runner_total_ms  = $s.base_worker_or_runner_total_ms
        benchmark_duration_s_bench      = if ($bench) { $bench.benchmark_duration_s } else { $null }
        request_throughput_req_s_bench  = if ($bench) { $bench.request_throughput_req_s } else { $null }
        output_tps_bench                = if ($bench) { $bench.output_token_throughput_tok_s } else { $null }
        total_tps_bench                 = if ($bench) { $bench.total_token_throughput_tok_s } else { $null }
        mean_ttft_ms_bench              = if ($bench) { $bench.mean_ttft_ms } else { $null }
        median_ttft_ms_bench            = if ($bench) { $bench.median_ttft_ms } else { $null }
        p99_ttft_ms_bench               = if ($bench) { $bench.p99_ttft_ms } else { $null }
        warning_count_server_log        = if ($issue) { $issue.warning_count } else { $null }
        error_count_server_log          = if ($issue) { $issue.error_count } else { $null }
        traceback_count_server_log      = if ($issue) { $issue.traceback_count } else { $null }
    }) | Out-Null
}
$masterRows | Sort-Object data_group, suite, @{Expression = { [int]($_.k) }} | Export-Csv -LiteralPath $masterRunPath -NoTypeInformation -Encoding UTF8

$groupStats = $fileInventory | Group-Object data_group | ForEach-Object {
    $g = $_.Group
    [pscustomobject]@{
        data_group       = $_.Name
        file_count       = $g.Count
        total_size_bytes = ($g | Measure-Object size_bytes -Sum).Sum
        log_count        = @($g | Where-Object { $_.extension -eq ".log" }).Count
        csv_count        = @($g | Where-Object { $_.extension -eq ".csv" }).Count
    }
}

$summaryMdPath = Join-Path $outputDir "TABLE_SUMMARY.md"
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# MQA FastPath Data Extraction Summary") | Out-Null
$md.Add("") | Out-Null
$md.Add("Generated at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')") | Out-Null
$md.Add("Output directory: $outputDir") | Out-Null
$md.Add("") | Out-Null
$md.Add("## File Overview") | Out-Null
$md.Add("| data_group | file_count | log_count | csv_count | total_size_bytes |") | Out-Null
$md.Add("|---|---:|---:|---:|---:|") | Out-Null
foreach ($gs in $groupStats | Sort-Object data_group) {
    $md.Add("| $($gs.data_group) | $($gs.file_count) | $($gs.log_count) | $($gs.csv_count) | $($gs.total_size_bytes) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Summary Rows (from summary.csv)") | Out-Null
$md.Add("| data_group | suite | k | status | throughput_tps | mean_tpot_ms | mean_itl_ms | mqa_mode | supports_gpu_multi_step | scoring_time_ms | verification_time_ms |") | Out-Null
$md.Add("|---|---|---:|---|---:|---:|---:|---|---|---:|---:|") | Out-Null
foreach ($s in $summaryRows | Sort-Object data_group, suite, @{Expression = { [int]($_.k) }}) {
    $md.Add("| $($s.data_group) | $($s.suite) | $($s.k) | $($s.status) | $($s.throughput_tps) | $($s.mean_tpot_ms) | $($s.mean_itl_ms) | $($s.mqa_mode) | $($s.supports_gpu_multi_step) | $($s.scoring_time_ms) | $($s.verification_time_ms) |") | Out-Null
}
$md.Add("") | Out-Null
$md.Add("## Output Tables") | Out-Null
$md.Add("| file | description |") | Out-Null
$md.Add("|---|---|") | Out-Null
$md.Add("| file_inventory.csv | All files with metadata, line count and SHA256 |") | Out-Null
$md.Add("| summary_combined.csv | Merged rows from all summary.csv files |") | Out-Null
$md.Add("| master_run_table.csv | Run-level merged table (summary + bench + server issue stats) |") | Out-Null
$md.Add("| meta_probe_kv.csv | Parsed key/value pairs from META and PROBE logs |") | Out-Null
$md.Add("| bench_metrics.csv | Parsed benchmark result metrics from bench logs |") | Out-Null
$md.Add("| log_issue_stats.csv | WARNING/ERROR/Traceback/SERVER_FAIL counts per log file |") | Out-Null
$md.Add("| log_key_values.csv | All key=value pairs extracted from all log lines |") | Out-Null
$md.Add("| log_last_values.csv | Last observed value for each key in each log file |") | Out-Null
$md.Add("| log_numeric_stats.csv | Numeric stats (count/min/max/avg/last) per key and file |") | Out-Null
$md.Add("| all_log_lines.csv | Full raw log lines with source path and line number |") | Out-Null

Set-Content -LiteralPath $summaryMdPath -Value $md -Encoding UTF8

Write-Output "Extraction completed."
Write-Output "Output directory: $outputDir"
Write-Output "Main summary doc: $summaryMdPath"
