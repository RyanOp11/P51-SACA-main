"""
SACA classification pipeline.

Attempts symptom extraction first.
If extraction fails, times out, or returns empty text,
the original raw symptom text is passed directly to
LogisticRegressionClassifier.classify().
"""
from classifier.logistic_regression import LogisticRegressionClassifier
from classifier.red_flags import check_red_flags, apply_floor


class SACAPipeline:

    def __init__(self, extractor=None):
        self.classifier = LogisticRegressionClassifier()
        self.extractor = extractor

    def _apply_red_flags(self, result: dict, classified_text: str, raw_text: str) -> dict:
        """Raise severity to the red-flag floor if danger signs are present (never lowers it)."""
        floor, reasons = check_red_flags(classified_text, raw_text)
        result["model_severity"] = result["severity"]
        result["model_confidence"] = result.get("confidence")
        result["red_flags"] = reasons
        result["override_applied"] = False

        final = apply_floor(result["severity"], floor)
        if final != result["severity"]:
            model_sev = result["severity"]
            new = self.classifier.build_result(
                severity=final,
                symptoms=result["symptoms"],
                reason=(f"Red flag: {', '.join(reasons)} -> at least {floor}. "
                        f"(Model alone predicted {model_sev}.)"),
                confidence=None,  # rules have no probability; don't reuse the model's
            )
            result.update(new)
            result["override_applied"] = True
        return result

    def classify(self, raw_text: str) -> dict:

        if not raw_text or not raw_text.strip():
            return self.classifier.classify(raw_text)

        if self.extractor is not None:

            try:
                extracted_text = self.extractor(raw_text)

                if extracted_text and extracted_text.strip():

                    print("[Pipeline] Symptom extraction succeeded.")
                    print(f"[Pipeline] Extracted: {extracted_text}")

                    result = self.classifier.classify(
                        extracted_text
                    )

                    result = self._apply_red_flags(result, extracted_text, raw_text)
                    result["input_source"] = "LLM extraction"
                    result["extracted_text"] = extracted_text
                    result["fallback_used"] = False

                    return result

                print("[Pipeline] Extraction returned empty text.")

            except TimeoutError:
                print("[Pipeline] LLM extraction timed out.")

            except Exception as e:
                print(f"[Pipeline] LLM extraction failed: {e}")

        print("[Pipeline] Using raw symptom text fallback.")

        result = self.classifier.classify(raw_text)
        result = self._apply_red_flags(result, raw_text, raw_text)

        result["input_source"] = "Raw text fallback"
        result["extracted_text"] = None
        result["fallback_used"] = True

        return result