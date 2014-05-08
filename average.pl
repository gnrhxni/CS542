#!/usr/bin/perl -w

use Time::HiRes;
use Data::Dumper;

my @files;
foreach (@ARGV) {
  my $fh;
  open($fh,$_) || die "bad $_";
  push(@files, $fh);
}


getdata: while (1) {
  foreach $fh (@files) {
    my $x = <$fh>;
    if ($x) { 
      @d = split(/\s+/,$x);
      $means->{$d[0]}->{PH} += $d[1];
      $means->{$d[0]}->{ST} += $d[2];
      $means->{$d[0]}->{COUNT} += 1;
    }
    else { last getdata; }
  }
}

foreach $numwords (sort { $a <=> $b } keys %{$means}) {
  if (0 == $means->{$numwords}->{COUNT}) { print "$numwords 0 0\n"; next; }
  $ph = $means->{$numwords}->{PH} / $means->{$numwords}->{COUNT};
  $st = $means->{$numwords}->{ST} / $means->{$numwords}->{COUNT};
  printf("%d %.3f %.3f\n", $numwords, $ph, $st);
}
