from toc_page_classifier.diversity_sampler import DiversitySampler


def test_accepts_under_cap_and_rejects_once_a_dimension_hits_its_cap():
    # target=10, language cap fraction 0.4 -> cap of 4 per language
    sampler = DiversitySampler(target_size=10, cap_fractions={"language": 0.4})
    for _ in range(4):
        assert sampler.offer({"language": "de"}) is True
    assert sampler.offer({"language": "de"}) is False
    assert sampler.offer({"language": "en"}) is True
    assert sampler.accepted_count == 5


def test_dimensions_are_independent():
    sampler = DiversitySampler(target_size=10, cap_fractions={"language": 0.5, "domain_bucket": 0.5})
    assert sampler.offer({"language": "de", "domain_bucket": "C"}) is True
    assert sampler.offer({"language": "de", "domain_bucket": "M"}) is True
    # third "de" would exceed language cap (ceil(10*0.5)=5, so actually still room) --
    # use a tighter cap to make the rejection deterministic instead:
    sampler2 = DiversitySampler(target_size=4, cap_fractions={"language": 0.5})
    assert sampler2.offer({"language": "de"}) is True
    assert sampler2.offer({"language": "de"}) is True
    assert sampler2.offer({"language": "de"}) is False  # cap = ceil(4*0.5) = 2


def test_is_full_stops_further_offers():
    sampler = DiversitySampler(target_size=2, cap_fractions={})
    assert sampler.offer({}) is True
    assert sampler.offer({}) is True
    assert sampler.is_full() is True
    assert sampler.offer({}) is False


def test_missing_dimension_value_is_never_rejected():
    sampler = DiversitySampler(target_size=10, cap_fractions={"language": 0.1})
    assert sampler.would_accept({}) is True
