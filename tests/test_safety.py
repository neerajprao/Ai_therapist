import pytest

from soulsync import safety

POSITIVE = [
    "I want to kill myself",
    "i wanna kill myself",
    "I'm going to end my life",
    "Im thinking about suicide",
    "I've been feeling suicidal lately",
    "I don't want to live anymore",
    "I dont want to be alive",
    "I can't go on like this",
    "Sometimes I wish I was dead",
    "Everyone would be better off without me",
    "I keep hurting myself",
    "I've been cutting myself again",
    "I think about self-harm a lot",
    "There's no reason to live",
    "I want to die",
    "I’m gonna end it all tonight",
    "I might take all my pills",
    "I thought about jumping, like jump off the bridge",
    "KILL MYSELF.",
    "life is not worth living",
    "Everyone would be fine if I just disappeared",
    "They'd all be better off if I was gone",
    "I just want to disappear forever",
    "No one would miss me",
]

NEGATIVE = [
    "This homework is killing me",
    "I'm dying to see that movie",
    "I'm dead tired after the gym",
    "I killed it in my presentation today",
    "My grandmother died last year and I miss her",
    "My phone died during the call",
    "I cut myself shaving this morning",
    "I want to live closer to my family",
    "I can't go on the trip next week",
    "I feel sad and lonely today",
    "Work has been really stressful",
    "I had a fight with my friend",
    "I want to end this argument with my brother",
    "The deadline is going to be the death of me",
    "I wish I could disappear to the mountains for a week",
    "",
]


@pytest.mark.parametrize("text", POSITIVE)
def test_crisis_language_is_detected(text):
    assert safety.is_crisis(text)


@pytest.mark.parametrize("text", NEGATIVE)
def test_everyday_language_is_not_flagged(text):
    assert not safety.is_crisis(text)


@pytest.mark.parametrize("text", [
    "Should I stop taking my anxiety medication?",
    "Can I increase my antidepressant dose on my own",
    "Is it safe to skip my meds for a few days",
    "Do you think I have depression?",
    "Can you diagnose me",
])
def test_medication_and_diagnosis_questions_get_a_referral(text):
    assert safety.needs_boundary_note(text)


@pytest.mark.parametrize("text", [
    "I had a stressful day at work",
    "My mom takes her medication every morning and she's doing well",
    "I need to stop procrastinating",
])
def test_ordinary_messages_get_no_referral(text):
    assert not safety.needs_boundary_note(text)


def test_normalize_handles_case_quotes_and_contractions():
    assert safety.normalize("I’M  Gonna, like... DO it!") == "i am going to like do it"
    assert safety.normalize("can't / cannot / don't") == "can not can not do not"


def test_crisis_message_contains_helpline_and_emergency_number():
    assert "14416" in safety.CRISIS_MESSAGE
    assert safety.EMERGENCY_NUMBER in safety.CRISIS_MESSAGE
    assert "AI" in safety.CRISIS_MESSAGE
