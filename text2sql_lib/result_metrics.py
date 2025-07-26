import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

from complexity_map import complexity_map      #  your external dict


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line := line.strip():
                yield json.loads(line)


def _ids_if(rows: Iterable[Dict], pred) -> List[int]:
    return [r["example_id"] for r in rows if pred(r)]


# --------------------------------------------------------------------------- #
# container
# --------------------------------------------------------------------------- #
@dataclass
class LLMResultMetrics:
    model_name: str
    total_queries: int

    # correctness buckets
    exact_match_ids: List[int] = field(default_factory=list)
    ast_match_ids: List[int] = field(default_factory=list)
    equiv_ids: List[int] = field(default_factory=list)        # llm_equivalent
    fulfills_intent_ids: List[int] = field(default_factory=list)
    superior_ids: List[int] = field(default_factory=list)
    equal_ids: List[int] = field(default_factory=list)
    incorrect_ids: List[int] = field(default_factory=list)

    # token / latency
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_latency_sec: float = 0.0

    # prices/meta
    in_price: float = 0.0
    out_price: float = 0.0
    open_source: bool = True
    reasoning: bool = False

    # buckets – filled automatically
    per_complexity_counts: Dict[str, Counter] = field(default_factory=dict)
    per_sql_complexity_counts: Dict[str, Counter] = field(default_factory=dict)
    _complexity_id_map: Dict[str, set] = field(default_factory=dict, repr=False)
    _sqlclass_id_map:   Dict[str, set] = field(default_factory=dict, repr=False)

    # --------------------------------------------------------------------- #
    # generic accuracy helper
    # --------------------------------------------------------------------- #
    def _accuracy(self, metric: str, bucket: Optional[str] = None) -> float:
            id_pool = {
                "exact":      self.exact_match_ids,
                "ast":        self.ast_match_ids,
                "technical":  self.equiv_ids,
                "intent":     set(self.fulfills_intent_ids + self.superior_ids), # + self.equal_ids,
            }[metric]

            good_ids: set = set(id_pool)          # ← ensure *set* for “&”

            # ─────────────────────────  global  ───────────────────────── #
            if bucket is None:
                return len(good_ids) / self.total_queries if self.total_queries else 0.0

            # ─────────────────────── bucketed path ────────────────────── #
            bucket_lc = bucket.lower()
            if bucket_lc in self._complexity_id_map:          # easy / medium / hard
                denom_ids = self._complexity_id_map[bucket_lc]
            else:                                             # sql_complexity bucket
                denom_ids = self._sqlclass_id_map.get(bucket, set())

            if not denom_ids:
                return 0.0
            return len(good_ids & denom_ids) / len(denom_ids)

    # public helpers ------------------------------------------------------ #
    def accuracy_exact(self, bucket: Optional[str] = None) -> float:
        return self._accuracy("exact", bucket)

    def accuracy_ast(self, bucket: Optional[str] = None) -> float:
        return self._accuracy("ast", bucket)

    def accuracy_judge_technical(self, bucket: Optional[str] = None) -> float:
        return self._accuracy("technical", bucket)

    def accuracy_judge_intent(self, bucket: Optional[str] = None) -> float:
        return self._accuracy("intent", bucket)

    # legacy (kept for quick printing) ------------------------------------ #
    @property
    def accuracy_exact_overall(self) -> float:
        return self.accuracy_exact()
    
    # superior
    @property
    def superior(self) -> float:
        return len(self.superior_ids) / self.total_queries if self.total_queries else 0
    
    @property
    def equal(self) -> float:
        return len(self.equal_ids) / self.total_queries if self.total_queries else 0


    # tokens / cost ------------------------------------------------------- #
    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def cost_eur(self) -> float:
        return (
            (self.total_prompt_tokens / 1_000_000) * self.in_price +
            (self.total_completion_tokens / 1_000_000) * self.out_price
        )

    # --------------------------------------------------------------------- #
    # string view
    # --------------------------------------------------------------------- #
    def __str__(self) -> str:
        return (
            f"{self.model_name}: "
            f"Exact {self.accuracy_exact():.2%} | "
            f"AST {self.accuracy_ast():.2%} | "
            f"Tech {self.accuracy_judge_technical():.2%} | "
            f"Intent {self.accuracy_judge_intent():.2%} | "
            f"Ø {self.avg_latency_sec:.2f}s | "
            f"Tokens {self.total_tokens:,} | "
            f"Cost €{self.cost_eur:.2f}"
        )

    # --------------------------------------------------------------------- #
    # factory
    # --------------------------------------------------------------------- #
    @classmethod
    def from_jsonl(
        cls,
        file: str,
        *,
        model_name="model",
        sql_complex_key="sql_complexity",
        in_price=0.0,
        out_price=0.0,
        open_source=True,
        reasoning=False,
        complexity_map: Optional[Dict[int, str]] = None,
    ):
        p = Path(file)
        rows = list(_read_jsonl(p))
        total = len(rows)

        # correctness buckets ------------------------------------------- #
        exact_ids = _ids_if(rows, lambda r: r.get("match_exact"))
        ast_ids   = _ids_if(rows, lambda r: r.get("match_ast"))
        equiv_ids = _ids_if(rows, lambda r: r.get("llm_equivalent"))
        fulfills  = _ids_if(
            rows,
            lambda r: (ej := r.get("enhanced_judgment")) and
                      ej.get("overall_assessment") in {"correct", "differently_correct"}
        )
        superior  = _ids_if(
            rows,
            lambda r: (ej := r.get("enhanced_judgment")) and
                      ej.get("superiority") in {"generated"}
        )
        equal  = _ids_if(
            rows,
            lambda r: (ej := r.get("enhanced_judgment")) and
                      ej.get("superiority") in {"equal"}
        )
        incorrect = [
            r["example_id"] for r in rows
            if r["example_id"] not in (*exact_ids, *ast_ids, *equiv_ids, *fulfills)
        ]
        
        # tokens / latency --------------------------------------------- #
        prompt_tok = sum(r.get("tokens_prompt", 0) for r in rows)
        compl_tok  = sum(r.get("tokens_completion", 0) for r in rows)
        avg_lat    = mean(r.get("latency_sec", 0.0) for r in rows) if rows else 0.0

        # complexity & sql buckets ------------------------------------- #
        per_comp: Dict[str, Counter] = defaultdict(Counter)
        comp_id_map: Dict[str, set]  = defaultdict(set)

        if complexity_map:
            for r in rows:
                eid     = r["example_id"]
                bucket  = complexity_map.get(eid, "unknown").lower()
                per_comp[bucket]["total"] += 1
                comp_id_map[bucket].add(eid)
                if eid in exact_ids:
                    per_comp[bucket]["exact"] += 1

        per_sql : Dict[str, Counter] = defaultdict(Counter)
        sql_id_map: Dict[str, set]   = defaultdict(set)
        for r in rows:
            eid = r["example_id"]
            sql_cls = r.get(sql_complex_key)
            if not sql_cls:
                continue
            per_sql[sql_cls]["total"] += 1
            sql_id_map[sql_cls].add(eid)
            if eid in exact_ids:
                per_sql[sql_cls]["exact"] += 1

        # ---------------------------------------------------------------- #
        return cls(
            model_name=model_name,
            total_queries=total,
            exact_match_ids=exact_ids,
            ast_match_ids=ast_ids,
            equiv_ids=equiv_ids,
            fulfills_intent_ids=fulfills,
            superior_ids=superior,
            equal_ids=equal,
            incorrect_ids=incorrect,
            total_prompt_tokens=prompt_tok,
            total_completion_tokens=compl_tok,
            avg_latency_sec=avg_lat,
            in_price=in_price,
            out_price=out_price,
            open_source=open_source,
            reasoning=reasoning,
            per_complexity_counts=per_comp,
            per_sql_complexity_counts=per_sql,
            _complexity_id_map=comp_id_map,
            _sqlclass_id_map=sql_id_map,
        )