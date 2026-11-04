"""A short request either executes or asks a bounded plain-language question."""
import unittest
from unittest.mock import patch
import test_agent_scale as fixture
from test_agent_scale import contact
from unicompose3d.schemas import AgentQuestion

class ClarificationTests(unittest.TestCase):
    setUp=fixture.AgentLoopTests.setUp
    tearDown=fixture.AgentLoopTests.tearDown
    prepare=fixture.AgentLoopTests.prepare
    evaluate=fixture.AgentLoopTests.evaluate
    def question(self):
        p=contact();p.program=None
        p.clarification=AgentQuestion(question="Which target part do you want?",options=["First part","Second part"])
        return p
    def test_question_pauses_before_geometry_and_answer_preserves_original_intent(self):
        self.llm.plan.side_effect=[self.question(),contact()]
        first=self.prepare()
        self.assertEqual(first["status"],"needs_input");self.assertEqual(first["candidates"],[])
        self.llm.evaluate.assert_not_called()
        answer={"clarification_of":first["id"],"clarification_answer":"Second part"}
        second=self.prepare(**answer)
        self.assertEqual(second["instruction"],self.body["instruction"])
        self.assertEqual(second["api_calls"]["plan"],2)
        sent=self.llm.plan.call_args.args[0]["user_answers"]
        self.assertEqual(sent[0]["answer"],"Second part")
        self.assertTrue(second["candidates"][0]["eligible"])
        third=self.prepare(**answer,refinement_of=second["id"])
        self.assertEqual(third["api_calls"]["plan"],1)
    def test_answer_cannot_be_reused_after_the_objects_or_instruction_change(self):
        self.llm.plan.return_value=self.question();question=self.prepare()
        response=self.client.post("/api/compositions",json=self.body|{
            "clarification_of":question["id"],"clarification_answer":"First part","instruction":"another goal"})
        self.assertEqual(response.status_code,400)
    def test_short_sufficient_instruction_does_not_require_an_extra_question_step(self):
        result=self.prepare(instruction="Put B on A")
        self.assertNotEqual(result["status"],"needs_input")
        self.assertEqual(self.llm.plan.call_count,1)
    def test_quick_mode_question_is_not_billed_twice_without_cache(self):
        self.llm.plan.return_value=self.question()
        response=self.client.post("/api/compose",json=self.body|{"use_cache":False})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()["status"],"needs_input")
        self.assertEqual(self.llm.plan.call_count,1)
        self.assertEqual(response.json()["api_calls"]["plan"],1)
    def test_geometry_repair_can_pause_to_ask_without_a_fake_placement(self):
        self.llm.plan.side_effect=[contact(),self.question(),contact()]
        with patch("unicompose3d.api.composition.build_candidates",return_value=([],{"feature_error":"target ambiguous"})):
            result=self.prepare(use_cache=False)
        self.assertEqual(result["status"],"needs_input")
        self.assertEqual(result["api_calls"]["plan"],2)
        self.assertEqual(result["candidates"],[])
        self.assertTrue(result["goal_contract"]["contact"])
        resumed=self.prepare(clarification_of=result["id"],clarification_answer="First part")
        self.assertTrue(resumed["candidates"][0]["eligible"])
        self.assertEqual(resumed["api_calls"]["plan"],3)
    def test_duplicate_or_excessive_options_are_not_valid_questions(self):
        with self.assertRaises(ValueError):AgentQuestion(question="Which one?",options=["Same","Same"])
        with self.assertRaises(ValueError):AgentQuestion(question="Which one?",options=["1","2","3","4","5"])
    def test_provider_prompt_explains_short_requests_and_null_program_exception(self):
        from unicompose3d.planning_prompt import PROMPT
        self.assertIn("program=null",PROMPT)
        self.assertIn("2-4 plain",PROMPT)
        self.assertNotIn("A null program is invalid.",PROMPT)
        self.assertIn("closed end",PROMPT)
if __name__=="__main__":unittest.main()
