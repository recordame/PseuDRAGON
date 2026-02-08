import os
from datetime import datetime
from typing import Any, Dict


class ReportGenerator:

    @staticmethod
    def ensure_output_dir():
        # Use the centralized output directory from config
        from config.config import DirectoryConfig

        output_dir = DirectoryConfig.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def generate_integrated_report(table_name: str, pii_analysis: Dict[str, Any], policy: Any, timestamp: str = None, output_dir: str = None, ) -> str:
        """
        Generate integrated report combining Stage 1 (PII Detection) and Stage 2 (Policy Synthesis)
        Stage 1 (PII 탐지)과 Stage 2 (정책 합성)를 결합한 통합 보고서 생성

        Args:
            table_name: Name of the table / 테이블 이름
            pii_analysis: PII analysis results from Stage 1 / Stage 1의 PII 분석 결과
            policy: Policy synthesis results from Stage 2 / Stage 2의 정책 합성 결과
            timestamp: Optional timestamp for filename / 파일명에 사용할 타임스탬프
            output_dir: Optional directory to save the report / 보고서를 저장할 선택적 디렉토리

        Returns:
            Path to the generated integrated report file / 생성된 통합 보고서 파일의 경로
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"integrated_pseudonymization_report_{table_name}.md")
        else:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = ReportGenerator.ensure_output_dir()
            filename = os.path.join(output_dir, f"integrated_pseudonymization_report_{table_name}_{timestamp}.md", )

        report_lines = [f"# Integrated Pseudonymization Report", f"", f"**Table Name:** {table_name}", f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"", f"---", f"", ]

        # ===== STAGE 1: PII DETECTION SECTION =====
        report_lines.extend([f"## Stage 1: PII Detection", f"", ])

        pii_count = sum(1 for col in pii_analysis.values() if col.get("is_pii", False))
        total_count = len(pii_analysis)

        report_lines.extend(
            [
                f"### Summary",
                f"",
                f"- **Total Columns:** {total_count}",
                f"- **PII Columns Detected:** {pii_count}",
                f"- **Non-PII Columns:** {total_count - pii_count}",
                f"",
                f"### Column-by-Column Analysis",
                f"",
            ]
        )

        for column_name, analysis in pii_analysis.items():
            is_pii = analysis.get("is_pii", False)
            pii_type = analysis.get("pii_type", "Unknown")
            confidence = analysis.get("confidence", "N/A")

            # Get rationale
            rationale = analysis.get("reasoning", analysis.get("rationale", "No rationale provided"))

            evidence_source = analysis.get("evidence_source", "No evidence source")
            column_comment = analysis.get("column_comment", "")

            status_emoji = "🔴" if is_pii else "🟢"

            report_lines.extend([f"#### {status_emoji} Column: `{column_name}`", f"", ])

            # Add column comment if available
            if column_comment:
                report_lines.extend([f"- **Description:** {column_comment}", ])

            report_lines.extend(
                [f"- **PII Status:** {'✅ PII Detected' if is_pii else '❌ Non-PII'}", f"- **PII Type:** {pii_type}", f"- **Confidence:** {confidence}", f"", f"**Rationale:**", f"> {rationale}", f"", ]
            )

            report_lines.extend([f"**Evidence Source:**", f"> {evidence_source}", f"", ])

        report_lines.extend([f"---", f"", ])

        # ===== STAGE 2: POLICY SYNTHESIS SECTION =====
        report_lines.extend([f"## Stage 2: Policy Synthesis", f"", ])

        if isinstance(policy, dict):
            pii_columns = list(policy.keys())

            report_lines.extend([f"### Summary", f"", f"- **Total PII Columns:** {len(pii_columns)}", f"- **Policies Generated:** {len(pii_columns)}", f"", f"### Pseudonymization Policies", f"", ])

            for column_name, col_policy in policy.items():
                pii_type = pii_analysis.get(column_name, {}).get("pii_type", "Unknown")
                column_comment = pii_analysis.get(column_name, {}).get("column_comment", "")
                methods = col_policy.get("recommended_methods", [])

                if not methods:
                    continue

                selected_method = methods[0]
                evidence_source = col_policy.get("evidence_source", "No evidence source")

                report_lines.extend([f"#### 🔐 Column: `{column_name}`", f"", ])

                # Add column comment if available
                if column_comment:
                    report_lines.extend([f"- **Description:** {column_comment}", ])

                report_lines.extend(
                    [
                        f"- **PII Type:** {pii_type}",
                        f"- **Selected Method:** `{selected_method.get('method', 'Unknown')}`",
                        f"- **Applicability:** {selected_method.get('applicability', 'N/A')}",
                        f"",
                        f"**Method Description:**",
                        f"> {selected_method.get('description', 'No description')}",
                        f"",
                        f"**Implementation Example:**",
                        f"> {selected_method.get('example_implementation', 'No example')}",
                        f"",
                        f"**Legal Evidence:**",
                        f"> {evidence_source}",
                        f"",
                    ]
                )

                if len(methods) > 1:
                    report_lines.extend([f"**Alternative Methods:**", f"", ])

                    for idx, candidate in enumerate(methods[1:], 1):
                        report_lines.extend(
                            [
                                f"{idx}. **{candidate.get('method', 'Unknown')}** (Applicability: {candidate.get('applicability', 'Unknown')})",
                                f"   - Description: {candidate.get('description', 'No description')}",
                                f"",
                            ]
                        )

                report_lines.extend([f"---", f"", ])
        else:
            # Handle Policy object
            pii_columns = [col for col, pol in policy.columns.items() if pol.is_pii]

            report_lines.extend(
                [
                    f"### Summary",
                    f"",
                    f"- **Preferred Method:** {getattr(policy, 'preferred_method', 'General Pseudonymization')}",
                    f"- **Total PII Columns:** {len(pii_columns)}",
                    f"- **Policies Generated:** {len(pii_columns)}",
                    f"",
                    f"### Pseudonymization Policies",
                    f"",
                ]
            )

            for column_name, col_policy in policy.columns.items():
                if not col_policy.is_pii:
                    continue

                action = col_policy.action
                pii_type = col_policy.pii_type
                column_comment = pii_analysis.get(column_name, {}).get("column_comment", "")

                report_lines.extend([f"#### 🔐 Column: `{column_name}`", f"", ])

                # Add column comment if available
                if column_comment:
                    report_lines.extend([f"- **Description:** {column_comment}", ])

                report_lines.extend(
                    [f"- **PII Type:** {pii_type}", f"- **Selected Method:** `{action.action.value}`", f"", f"**Method Description:**", f"> {action.rationale}", f"", f"**Parameters:**", ]
                )

                if action.parameters:
                    for key, value in action.parameters.items():
                        report_lines.append(f"- `{key}`: {value}")
                else:
                    report_lines.append(f"> No additional parameters")

                report_lines.extend([f"", f"**Legal Evidence:**", f"> {action.legal_evidence}", f"", ])

                if (col_policy.candidate_actions and len(col_policy.candidate_actions) > 1):
                    report_lines.extend([f"**Alternative Methods:**", f"", ])

                    for idx, candidate in enumerate(col_policy.candidate_actions[1:], 1):
                        report_lines.extend([f"{idx}. **{candidate.action.value}**", f"   - Description: {candidate.rationale}", f"   - Evidence: {candidate.legal_evidence}", f"", ])

                report_lines.extend([f"---", f"", ])

        # Footer
        report_lines.extend(
            [f"", f"---", f"", f"**Report Generated by PseuDRAGON Framework**", f"", f"This integrated report combines PII detection results and pseudonymization policy recommendations.", ]
        )

        report_content = "\n".join(report_lines)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)

        return filename
